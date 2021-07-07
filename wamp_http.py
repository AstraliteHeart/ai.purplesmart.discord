import asyncio
import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
from random import randint

import aiohttp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
SECRET = os.getenv("CROSSBAR_SECRET")


def _compute_signature(body, key, secret, sequence):
    """
    Computes the signature.
    Described at:
    http://crossbar.io/docs/HTTP-Bridge-Services-Caller/
    Reference code is at:
    https://github.com/crossbario/crossbar/blob/master/crossbar/adapter/rest/common.py
    :return: (signature, none, timestamp)
    """

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    nonce = str(randint(0, 2 ** 53))

    # Compute signature: HMAC[SHA256]_{secret} (key | timestamp | seq | nonce | body) => signature
    hm = hmac.new(bytes(secret, "utf-8"), None, hashlib.sha256)
    hm.update(bytes(key, "utf-8"))
    hm.update(bytes(timestamp, "utf-8"))
    hm.update(bytes(sequence, "utf-8"))
    hm.update(bytes(nonce, "utf-8"))
    hm.update(bytes(body, "utf-8"))

    signature = base64.urlsafe_b64encode(hm.digest())
    return signature, nonce, timestamp


class NoBackendAvailable(Exception):
    pass


class UnknownError(Exception):
    pass


async def request(path, data):
    timeout = aiohttp.ClientTimeout(sock_connect=15, connect=15, total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = {"procedure": path, "args": [data]}
            encoded_params = json.dumps(data)

            sequence = str("0")
            signature, nonce, timestamp = _compute_signature(
                encoded_params, "caller", SECRET, sequence
            )
            params = {
                "timestamp": timestamp,
                "seq": sequence,
                "nonce": nonce,
                "signature": signature.decode("utf-8"),
                "key": "caller",
            }
            logger.info("posting data")
            async with session.post(
                "http://pone.dev:8080/call",
                data=encoded_params,
                params=params,
                headers={"content-type": "application/json"},
            ) as resp:
                resp = await resp.json()
                if "error" in resp.keys():
                    error = resp["error"]
                    if error == "wamp.error.no_such_procedure":
                        raise NoBackendAvailable(
                            "Sorry, no backends currently active for this model."
                        )
                    else:
                        raise UnknownError("Unknown error: %s" % error)

                return resp

    except asyncio.TimeoutError as e:
        raise UnknownError("Unknown timeout error: %s" % e)


if __name__ == "__main__":

    async def test_request():
        print(
            await request(
                "com.purplesmart.router.api",
                {
                    "method": "tts/v1",
                    "query": 'Twilight Sparkle said " "',
                },
            )
        )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_request())
    loop.close()
