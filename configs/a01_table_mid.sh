# A01 coffee table, generated from a lower view rendered out of the 3DGS (novel_view.py: 0.22 above the floor, 0.6 out):
# from the panorama centre SAM 3D only made a flat disc without the base (4 seeds). Skip crop / align: the view exists.
RUN=A01_table_mid; NAME=table; SCENE=A01/rep1; PLY=filled_02.ply
YAW=0; PITCH=0; FOV=60
PROMPT="coffee table"; PARTS=(); REST=floor
EXTRA_RUNS=(A01_table)       # the panorama crop of the same table: the sharpest view, joins the vote and the fit
