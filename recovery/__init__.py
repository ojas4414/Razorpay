"""The revenue-recovery pipeline: detect, intervene, measure, audit.

Each stage reads the same observable session state the engine emits. Nothing
here can see core.friction's latent intent or price sensitivity -- those decide
whether a session is abandoned, and a detector that could read them would be an
oracle rather than a detector.
"""
