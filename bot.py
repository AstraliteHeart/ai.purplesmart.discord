import io
import logging
import os
import textwrap
from base64 import b64decode
from timeit import default_timer as timer

import coloredlogs
import discord
import humanize
import nltk
from discord.ext import commands
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from pathvalidate import sanitize_filename
from profanity_check import predict_prob

from wamp_http import NoBackendAvailable
from wamp_http import request as wamp_request

CHANNELS = {"nsfw": "MLP|NSFW|", "sfw": "MLP|SFW|"}
MIN_CHARACTERS = 150
MAX_CHARACTERS = 700
TEXT_MIN_CHARACTERS = 20
TEXT_MAX_CHARACTERS = 300


GPT2_DELAYS = {
    "Admin": 1,
    "Drone": 15,
    "Tech": 15,
    "Supporter": 10,
    "@everyone": 30,
}

VOICE_DELAYS = {
    "Admin": 1,
    "Drone": 5,
    "Tech": 5,
    "Supporter": 10,
    "@everyone": 30,
}

GPT2_DM_ENABLED = {
    "Admin": True,
    "Drone": True,
    "Tech": True,
    "Supporter": False,
    "@everyone": False,
}

GPT2_NSFW_ENABLED = {
    "Admin": True,
    "Drone": True,
    "Tech": True,
    "Supporter": False,
    "@everyone": False,
}

GPT2_SELECT_MODEL = {
    "Admin": True,
    "Drone": False,
    "Tech": False,
    "Supporter": False,
    "@everyone": False,
}

MAX_MSG_LENGTH = 800

USER_STORY_TIME = dict()
USER_VOICE_TIME = dict()

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

load_dotenv()
token = os.getenv("DISCORD_TOKEN")

nltk.download("punkt")

bot = commands.Bot(command_prefix="!")
GUILD = None


def clear_expired():
    current_time = timer()
    for user in list(USER_STORY_TIME):
        last_story_time, timeout = USER_STORY_TIME[user]
        if current_time - last_story_time > timeout:
            del USER_STORY_TIME[user]


def clear_expired_voice():
    current_time = timer()
    for user in list(USER_VOICE_TIME):
        last_story_time, timeout = USER_VOICE_TIME[user]
        if current_time - last_story_time > timeout:
            del USER_VOICE_TIME[user]


@bot.event
async def on_ready():
    logger.info(f"{bot.user} has connected to Discord!")
    global GUILD
    GUILD = bot.get_guild(670866322619498507)


"""
@bot.event
async def on_message(message):
    if message.channel.name in ['sfw', 'nsfw']:
        await bot.process_commands(message)
"""


@bot.command(
    name="pstory",
    help="""
Same as !story but allows to set special generation params, use as "!pstory temperature=value&top_p=value&top_k=value&penalty=value text".

**Temperature** makes the text more diverse as it increases, may either generate interesting stories or devolve into total chaos. Default temperature is 0.85. Accepted values [0.1, 1.0], try a step of by 0.01.
**Top-k** selects tokens(i.e. words) from a list of K most likely tokens. Works good for some inputs but not so great for 'narrow distribution', i.e. where one token is much more likely to be used than any other. Accepted values [5, 40].
**Top-p** select tokens from a list cut of tokens above specific probability value P. Works good for 'narrow distribution' but may limit choices for 'broad distribution'. Accepted values [0.1, 1.0], try a step of by 0.01.
You can only set either top_p or top_k. By default top_p of 0.9 is used.
**Penalty** helps to stop model from looping (i.e. repeateg the same text over and over) on low temperatures. Accepted values [0.1, 1.0]

You can read more at https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277

If you DM the bot you can also provide **model** parameter to select between 'sfw' and 'nsfw' models.

Here are some good starting points.
  * temperature=0.85&top_p=0.9 - Good starting point, high temperature for diverse results with top-p sampling.
  * temperature=0.3&top_p=0.9&=penalty=0.85 - Same as above but prevent looping at lower temperature by using repetition penalty.
  * temperature=0.3&top_k=40&=penalty=0.85 - Now use top-k sampling, may or may not be better than top-k depending on input.  

There are no 'best' values that we've discovered so far, quality of story depends on these parameters but also length of user prompt and amount of model training. Please experiment and report back any good values!""",
)
async def pstory(ctx, *, prompt: str):
    data = {}
    params, text = prompt.split(" ", 1)
    if "=" in params:
        prompt = text
        data = dict(item.split("=") for item in params.split("&"))

    """
    try:
        settings = params.split("&")
        for setting in settings:
            name, value = setting.split("=")
            data[name] = value
    except e:
        await ctx.send("Sorry, can't parse the params.")
        return
    """
    print(data)

    # temperature = float(data.get("temperature", 0.85))
    temperature = data.get("temperature")
    # top_p = float(data.get("top_p", 0.85))
    top_p = float(data.get("top_p", 0))
    # top_k = int(data.get("top_k", 0))
    top_k = int(data.get("top_k", 0))
    penalty = float(data.get("penalty", 0))
    requested_model = data.get("model")

    if top_k > 0 and top_p > 0:
        await ctx.send("Sorry, can't set top_k and top_p at the same time.")
        return

    if "top_k" in data and (top_k < 5 or top_k > 40):
        await ctx.send("Sorry, wrong 'top_k' value, expected [5, 40]")
        return

    if temperature:
        temperature = float(temperature)
        if temperature < 0.1 or temperature > 1.0:
            await ctx.send("Sorry, wrong 'temperature' value, expected [0.1, 1.0]")
            return

    if "penalty" in data and (penalty < 0.1 or penalty > 1.0):
        await ctx.send("Sorry, wrong 'penalty' value, expected [0.1, 1.0]")
        return

    if "top_p" in data and (top_p < 0.1 or top_p > 1.0):
        await ctx.send("Sorry, wrong 'top_p' value, expected [0.1, 1.0]")
        return

    await gen_story(
        ctx,
        prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        penalty=penalty,
        requested_model=requested_model,
    )


@bot.command(
    name="say",
    help="Generate a voice clip using Twilight Sparkle's Tacotron2+WaveGlow model from Pony Preservation Project",
)
async def say(ctx: discord.ext.commands.Context, *, prompt: str):
    if len(prompt) < TEXT_MIN_CHARACTERS:
        response = (
            f"Sorry, prompt must be at least {TEXT_MIN_CHARACTERS} characters, got %s"
            % len(prompt)
        )
        await ctx.send(response)
        return

    if len(prompt) > TEXT_MAX_CHARACTERS:
        response = (
            f"Sorry, prompt must be less than {TEXT_MAX_CHARACTERS} characters, got %s"
            % len(prompt)
        )
        await ctx.send(response)
        return

    profanity_score = predict_prob([prompt])[0]
    if profanity_score > 0.3:
        response = f"Sorry, profanity score is too high: {profanity_score:.3f}"
        await ctx.send(response)
        return

    msg = f"<@{ctx.author.id}> Got it, working on a voice sample for you!"
    await ctx.send(msg)

    path = f"com.purplesmart.router.api"
    data = {"query": prompt, "method": "tts/v1", "kernel": None, "params": {}}
    try:
        resp = (await wamp_request(path, data))["args"][0]
    except NoBackendAvailable as e:
        logging.exception(f"Generation error: {e}")
        await ctx.send(f"{e}")
        return
    except Exception:
        response = "Sorry, I crashed!\n"
        logging.exception("Generation error")
        await ctx.send(response)
        return

    output = resp["output"]

    sound = io.BytesIO(b64decode(output["generated_data"]))
    # sound = io.BytesIO(resp["data"])
    filename = sanitize_filename(prompt).lower()[:40]
    file = discord.File(sound, filename="%s.ogg" % filename)
    embed = discord.Embed(title="A message from Twilight")

    arpabet = output["generated_text"] or "no arphabet data"
    embed.add_field(name="Arpabet", value=arpabet)
    await ctx.send(file=file, embed=embed)


@bot.command(name="story", help="Give me a prompt, get a story")
async def story(ctx, *, prompt: str):
    await gen_story(ctx, prompt)


SENTENCE_END = frozenset({".", "!", "?"})


def remove_last_sentence(text: str):
    if text[-1] in SENTENCE_END:
        return text

    paragraphs = text.split("\n\n")
    paragraphs = [p for p in paragraphs if p]
    last_paragraph = paragraphs.pop()
    last_paragraph = " ".join(sent_tokenize(last_paragraph)[:-1])
    paragraphs.append(last_paragraph)
    return "\n\n".join(paragraphs)


async def gen_story(
    ctx: discord.ext.commands.Context,
    prompt,
    temperature=None,
    top_p=None,
    top_k=None,
    penalty=None,
    requested_model=None,
):
    forced_model = None
    is_owner = await ctx.bot.is_owner(ctx.author)

    if ctx.guild is None:
        # Direct Message
        member = GUILD.get_member(ctx.author.id)
        if not member:
            response = (
                "Sorry, you need to be a member of the AskPonyAi channel to DM the bot."
            )
            await ctx.send(response)
            return

        roles = member.roles
    else:
        roles = ctx.author.roles

        can_select_model = max(
            filter(
                lambda x: x is not None,
                [GPT2_SELECT_MODEL.get(role.name) for role in roles],
            )
        )

        if requested_model is not None:
            if can_select_model:
                forced_model = requested_model
            else:
                response = "Sorry, you can only select the model when you DM."
                await ctx.send(response)
                return

    if ctx.guild is None:
        can_use_dm = max(
            filter(
                lambda x: x is not None,
                [GPT2_DM_ENABLED.get(role.name) for role in roles],
            )
        )
        if not is_owner and not can_use_dm:
            response = "Sorry, DM based story generation is not yet available."
            await ctx.send(response)
            return

    timeout_for_role = min(
        filter(lambda x: x is not None, [GPT2_DELAYS.get(role.name) for role in roles])
    )

    try:
        if len(prompt) < MIN_CHARACTERS:
            response = (
                f"Sorry, prompt must be at least {MIN_CHARACTERS} characters, got %s"
                % len(prompt)
            )
            await ctx.send(response)
            return

        if len(prompt) > MAX_CHARACTERS:
            response = (
                f"Sorry, prompt must be less than {MAX_CHARACTERS} characters, got %s"
                % len(prompt)
            )
            await ctx.send(response)
            return

        clear_expired()
        start = timer()

        last_story = None
        if ctx.author.id in USER_STORY_TIME:
            last_story, timeout = USER_STORY_TIME[ctx.author.id]

        if last_story and start - last_story < timeout:
            wait_for = humanize.naturaldelta(timeout - (start - last_story))
            await ctx.send(
                f"<@{ctx.author.id}> Cooldown active, please wait {wait_for}."
            )
            return

        profanity_score = predict_prob([prompt])[0]
        if profanity_score > 0.15 and str(ctx.channel) != "nsfw":
            response = f"Sorry, profanity score is too high: {profanity_score}"
            await ctx.send(response)
            return

        USER_STORY_TIME[ctx.author.id] = (start, timeout_for_role)

        data = {}
        if top_p and top_p > 0:
            data["top_p"] = top_p
        if top_k and top_k > 0:
            data["top_k"] = top_k
        if penalty and penalty > 0:
            data["penalize"] = penalty
        if temperature and temperature > 0:
            data["temperature"] = temperature
        data["max_length"] = 300

        if ctx.guild is None:
            if requested_model is not None:
                if requested_model not in CHANNELS.keys():
                    response = "Sorry, you can only select the model when you DM."
                    await ctx.send(response)
                    return
                else:
                    inference_model = CHANNELS[requested_model]
            else:
                inference_model = CHANNELS["sfw"]
        else:
            if forced_model:
                inference_model = CHANNELS.get(forced_model)
            else:
                inference_model = CHANNELS.get(str(ctx.channel))
            if not inference_model:
                response = "Sorry, I only operate in [%s] channels." % ", ".join(
                    [k for k in CHANNELS.keys()]
                )
                await ctx.send(response)
                return

        msg = f"<@{ctx.author.id}> Got it, working on a story for you!"
        await ctx.send(msg)

        path = f"com.purplesmart.router.api"
        print(data)
        data = {
            "query": inference_model + prompt,
            "method": "generate/v1",
            "kernel": None,
            "params": data,
        }
        resp = await wamp_request(path, data)
        print(resp)
        output = resp["args"][0]["output"]
        text = output[0]["generated_text"][len(inference_model) :]

        print("text", text)

        text = remove_last_sentence(text.strip()).strip()

        if len(text) == 0:
            response = "Sorry, I couldn't generate a story for this input"
            await ctx.send(response)
            return

        paragraphs = text.split("\n\n")

        end = timer()
        gen_seconds = end - start

        total_length = 0
        text = ""
        messages = []
        per_message_paragraphs = []

        split_paragraphs = []
        for p in paragraphs:
            if len(p) > MAX_MSG_LENGTH:
                pieces = textwrap.wrap(p, MAX_MSG_LENGTH - 25)
                for piece_id, piece in enumerate(pieces):
                    if piece_id != 0:
                        pieces[piece_id] = '...' + piece
                    
                    if piece_id != len(pieces) - 1:
                        pieces[piece_id] = piece + '...'

                split_paragraphs.extend(pieces)

                print ('split_paragraphs', split_paragraphs)
            else:
                split_paragraphs.append(p)

        for paragraph in split_paragraphs:
            total_length += len(paragraph)
            if total_length > MAX_MSG_LENGTH:
                messages.append("\n\n".join(per_message_paragraphs) + "\n\n")
                total_length = len(paragraph)
                per_message_paragraphs = [paragraph]
            else:
                per_message_paragraphs.append(paragraph)

        if per_message_paragraphs:
            messages.append("\n\n".join(per_message_paragraphs))

        print("messages", messages, paragraphs)

        embed = discord.Embed(title="A Pony AI story")
        embed.set_author(name=ctx.bot.user, icon_url=ctx.bot.user.avatar_url)
        embed.add_field(
            name=f"\u200B", value=f"Prompt by <@{ctx.author.id}>", inline=False
        )
        for msg in messages:
            embed.add_field(name=f"\u200B", value=msg, inline=False)

        embed.set_footer(
            text=(
                f"Generation time: {gen_seconds:.2f}. Profanity score {profanity_score:.2f}.\n"
                f"top_p: {data.get('top_p', 'N/A')}. top_k: {data.get('top_k', 'N/A')}. temperature: {data.get('temperature', 'N/A')}. penalty: {data.get('penalize', 'N/A')}. model: {inference_model}"
            )
        )
        await ctx.send(embed=embed)

    except NoBackendAvailable as e:
        logging.exception(f"Generation error: {e}")
        await ctx.send(f"{e}")
        return
    except Exception:
        response = "Sorry, I crashed!\n"
        logging.exception("Generation error")
        await ctx.send(response)
        return


if __name__ == "__main__":
    bot.run(token)
