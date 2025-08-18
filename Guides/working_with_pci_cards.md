# Working with PCI / PCI-E cards

If you're using internal cards, make sure that the correctly input is selected. 
You can use [v4l2-ctl](https://petersopus.wordpress.com/tag/v4l2/) to configure it, like this (using a Pinnacle 110i pci card):

```console
v4l2-ctl -i 3 -d 1 --set-standard=0
``` 

When

- -i 3: selecting 3rd input option, in this case, S-VIDEO
- -d 1: the input device (/dev/videoX)
- --set-standard: 0 is the NTSC option

You can put it on `/etc/rc.local` or set manually before start capture. 
