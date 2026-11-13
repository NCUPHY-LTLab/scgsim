"""Planning that only AEDT's native construction uses: surface, contact, domain and topology planning, evidence and tags.

In: the effective input from ``interpretation``.
Out: the surface, contact, domain and topology plans, the tag and evidence records that AEDT builds from, and the entity lookups and coordinate and bounds operations those plans share. The surface-partition and construction-body records live in ``surface_planning``, the only stage that reads them.
Next: ``scgsim.aedt``. Nothing on the Palace route (``scgsim.palace``, ``sgb``, ``surface_epr``) imports this folder.
"""
