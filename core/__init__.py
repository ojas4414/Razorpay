"""Domain logic shared by the HTTP services and the simulation engine.

Everything in this package is plain Python: no FastAPI, no Redis, no httpx, no
network. The microservices in backend/ are adapters that expose these rules over
HTTP; live/engine.py calls the same rules in-process.

That split exists for a concrete reason. The batch experiment runs thousands of
sessions, and routing each one through four HTTP hops would make reproducing the
headline numbers require docker-compose and take hours. Sharing the domain layer
instead means the simulation and the services cannot drift apart -- which they
had: simulate.py priced products over HTTP against a gateway route that does not
exist, so every price silently fell back to 0.0.
"""
