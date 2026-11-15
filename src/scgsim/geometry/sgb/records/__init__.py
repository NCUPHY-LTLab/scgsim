"""The records SGB's stages hand to one another, and what derives from a record alone.

In: the exact kernel (``scgsim.geometry.exact``); this folder imports no other part of SGB.
Out: the prism, footprint and catalogue records, cut families, band regions and support regions, the labels
arrangement cells receive and the interface type they give a face, and content-addressed ids.
Next: every stage of SGB imports it.

* ``prisms``       the prism, footprint and catalogue records
* ``families``     cut families, the record bands hand to the arrangement
* ``band_regions`` the band regions and required seams of an inset family
* ``support``      the support region of a surface partition
* ``labels``       the labels arrangement cells receive
* ``interfaces``   the interface type of a face from the labels (domains) on its two sides
* ``identity``     the content-addressed ids of the records
"""
