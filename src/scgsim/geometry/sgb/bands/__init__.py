"""S3: the inset bands a caller requests, as cut families.

In: the ``PrismCatalogue``, the typed ``InsetRequest`` values and the physical pre-pass (``LabelledArrangements`` of
the physical prisms, from ``arrangement``).
Out: the catalogue carrying one ``CurveFamilyRecord`` per band family, and the ``InsetPlan`` that says which faces were
banded.
Next: ``arrangement`` derives the arrangements again, the families noded together with the source curves.
"""
