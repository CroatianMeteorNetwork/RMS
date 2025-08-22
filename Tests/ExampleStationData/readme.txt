The file stations.tar.bz2 contains sample station data from stations at Perth Observatory.
The files are arranged in the same structure as used by the multi-cam linux project, and are intended for test purposes.

Extract the files using the command 

tar -xf stations.tar.bz2

And optionally, move the created folder with

mv Stations ~/source/

If you wish to experiment with SkyFit2

cd ~/source/RMS
source ~/vRMS/bin/activate
python -m Utils.SkyFit2 -c ~/source/Stations/AU000A/.config  ~/source/Stations/AU000A

or

python -m Utils.SkyFit2 -c ~/source/Stations/AU000C/.config  ~/source/Stations/AU000C

or

python -m Utils.SkyFit2 -c ~/source/Stations/AU000D/.config  ~/source/Stations/AU000D -m ~/source/Stations/

or

python -m Utils.SkyFit2 -c ~/source/Stations/AU000F/.config  ~/source/Stations/AU000F

or

python -m Utils.SkyFit2 -c ~/source/Stations/AU000G/.config  ~/source/Stations/AU000G

or

python -m Utils.SkyFit2 -c ~/source/Stations/AU000K/.config  ~/source/Stations/AU000K

Station AU000D has a mask covering some trees
All are 6mm lenses, except for AU000K, which is 16mm. 

