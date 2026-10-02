"""Compatibility name for the CtrlSpeech autoregressive backbone.

The Japanese branch uses LFM2.5-350M exclusively. New code should import
LFM2SpeechBackbone directly; CausalAR remains as an alias for older call sites.
"""

from ..backbone.lfm2 import LFM2SpeechBackbone


class CausalAR(LFM2SpeechBackbone):
    pass
