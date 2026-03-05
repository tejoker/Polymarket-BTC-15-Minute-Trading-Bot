"""
Fast ECDSA Pre-computation Module
Replaces standard slow deterministic ECDSA signing by pre-computing
nonces (k) and curve points (R) in a background thread.
When an order arrives, it only performs rapid modular arithmetic.
"""

import threading
import queue
import time
import os
import logging

try:
    import ecdsa
    from ecdsa.ellipticcurve import Point
    from ecdsa.curves import SECP256k1
except ImportError:
    ecdsa = None

logger = logging.getLogger(__name__)


class ECDSAFasterSigner:
    """
    Precomputes ECDSA nonces for SECP256k1.
    Pool entries: (k, k_inv, r, y_parity) where y_parity is 0 or 1.
    """
    def __init__(self, target_pool_size=100):
        self.target_pool_size = target_pool_size
        self.precomputed = queue.Queue(maxsize=target_pool_size)
        self.running = False
        self._thread = None

        if ecdsa:
            self.curve = SECP256k1
            self.n = self.curve.order
            self.G = self.curve.generator

    def start(self):
        if not ecdsa:
            logger.warning("ecdsa package not found, skipping precomputation")
            return

        self.running = True
        self._thread = threading.Thread(target=self._precompute_worker, daemon=True)
        self._thread.start()
        logger.info("Started ECDSA precomputation thread")

    def stop(self):
        self.running = False
        if self._thread:
            while not self.precomputed.empty():
                try:
                    self.precomputed.get_nowait()
                except queue.Empty:
                    break
            self._thread.join(timeout=1.0)

    def _precompute_worker(self):
        while self.running:
            try:
                k_bytes = os.urandom(32)
                k = int.from_bytes(k_bytes, 'big')
                if k >= self.n or k == 0:
                    continue

                # Expensive elliptic curve multiplication
                point = k * self.G
                r = point.x() % self.n
                if r == 0:
                    continue

                k_inv = pow(k, -1, self.n)
                y_parity = point.y() % 2  # 0 = even, 1 = odd

                self.precomputed.put((k, k_inv, r, y_parity))
            except Exception as e:
                logger.error(f"Error in ECDSA precomputation: {e}")
                time.sleep(1)

    def sign_hash(self, message_hash_bytes: bytes, private_key_int: int) -> tuple:
        """
        Produce a signature using a precomputed nonce.
        Returns: (v, r, s) where v is the Ethereum recovery id (27 or 28).
        """
        if not self.running or self.precomputed.empty():
            return None  # Caller should fall back to standard signer

        try:
            k, k_inv, r, y_parity = self.precomputed.get(timeout=0.05)
        except queue.Empty:
            return None

        z = int.from_bytes(message_hash_bytes, 'big')

        # Fast modular arithmetic (the only work at signing time)
        s = (k_inv * (z + r * private_key_int)) % self.n

        if s == 0:
            return None

        # Enforce low S (EIP-2)
        if s > self.n // 2:
            s = self.n - s
            y_parity ^= 1  # Flip parity when negating s

        v = 27 + y_parity
        return (v, r, s)

    @property
    def pool_size(self) -> int:
        return self.precomputed.qsize()


# Global instance
fast_signer = ECDSAFasterSigner()


def apply_ecdsa_patch():
    """
    Start the ECDSA precomputation background thread.

    The actual signing in py_clob_client happens via asyncio.to_thread
    (already off the event loop), so this provides incremental improvement
    by having nonces ready when signing starts.
    """
    if not ecdsa:
        logger.info("ecdsa package not found, ECDSA precomputation skipped")
        return False

    fast_signer.start()
    logger.info(
        f"ECDSA precomputation started (pool target: {fast_signer.target_pool_size})"
    )
    return True
