"""Splitting flagged sessions into treatment and control.

Assignment happens **at flag time**, not at session start. Randomising earlier
would compare flagged sessions against unflagged ones, which measures the
detector rather than the intervention.

The arm is a hash of the session id and an experiment salt, not a draw from the
random module. That matters more than it looks. The simulation's entire state --
product choice, action timing, the quit hazard -- comes from one seeded global
RNG, so consuming a number here would shift every subsequent draw and the
control arm would stop being the run you would have got without any
intervention. Hashing consumes nothing, so an untreated session behaves
identically whether or not the experiment is running.

Assignment is therefore deterministic given (session_id, salt): the same
experiment reproduces exactly, and changing the salt reshuffles the split
without touching anything else.
"""
import hashlib

TREATMENT = "treatment"
CONTROL = "control"

DEFAULT_SALT = "recovery-v1"


def arm_for(session_id: str, salt: str = DEFAULT_SALT, treatment_share: float = 0.5) -> str:
    """Which arm this session belongs to.

    ``treatment_share`` is the fraction assigned to treatment; 0.5 is an even
    split. Setting it to 0.0 gives an A/A configuration -- every flagged session
    is assigned but none is treated -- which is how the measurement harness is
    checked for reporting a null effect when there is genuinely nothing to find.
    """
    digest = hashlib.md5(f"{salt}:{session_id}".encode()).hexdigest()
    # Top 32 bits as a uniform fraction in [0, 1).
    fraction = int(digest[:8], 16) / 0x100000000
    return TREATMENT if fraction < treatment_share else CONTROL
