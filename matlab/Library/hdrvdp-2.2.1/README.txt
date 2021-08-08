HDR-VDP-2: A calibrated visual metric for visibility and quality
predictions in all luminance conditions

This directory contains matlab code of the HDR-VDP-2 - a visual
difference predictor for high dynamic range images. This is the
successor of the original HDR-VDP.

Always check for the latest release of the metric at:

http://hdrvdp.sourceforge.net/

The current version number and the list of changes can be found in the
ChangeLog.txt.

-----------------------------------------------------------------
To install the metric just add the hdrvdp directory to the matlab path. 

HDR-VDP-2 requres matlabPyrTools (http://www.cns.nyu.edu/~lcv/software.html).
The first invocation of the hdrvdp() function will add matlabPyrTools 
automatically to the matlab path. If you already have matlabPyrTools in 
the path, the metric may fail, as HDR-VDP-2 requires a patched version of 
that toolbox. 


-----------------------------------------------------------------
To run the metric:

Check Contents.m and the documentation for hdrvdp.m,
hdrvdp_visualize.m and hdrvdp_pix_per_deg.m


-----------------------------------------------------------------
Citations:

If you find this metric useful, please cite the paper:

HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions
Rafał Mantiuk, Kil Joong Kim, Allan G. Rempel and Wolfgang Heidrich.
In: ACM Transactions on Graphics (Proc. of SIGGRAPH'11), 30(4), article no. 40, 2011

AND the version of the metric you used, for example "HDR-VDP 2.1.1". Check
ChangeLog.txt for the current version of the metric.

-----------------------------------------------------------------
Contact:

If possible, please post your question to the google group:
http://groups.google.com/group/hdrvdp

If the communication needs to be confidential, contact me
directly. Please include "[n0t5pam]" in the subject line so that your
e-mail is not filtered out by the SPAM filter).

Rafal Mantiuk <mantiuk@gmail.com>

-----------------------------------------------------------------
For more more information, refer to the project web-site:
http://hdrvdp.sourceforge.net/
