Camera Parameters Dialog mappings  
=================================  
``` bash
Exposure Mode	Param.[0].ExposureParam.Level		0  
Min Time	Param.[0].ExposureParam.LeastTime	    '0x00009C40'  
Max Time	Param.[0].ExposureParam.MostTime	    '0x00009C40'  
DayNight Mode	Param.[0].DayNightColor			    '0x00000002'  
BLC		Param.[0].BLCMode			                '0x00000000'  
Auto Iris	Param.[0].ApertureMode			        '0x00000000'  
White Balance	Param.[0].WhiteBalance			    '0x00000002'  
AE Ref		Param.[0].ElecLevel			            100  
Image Style	ParamEx.[0].Style 			            'type1'  
DNC Thresh	Param.[0].DncThr			            50  
DWDR		ParamEx.[0].BroadTrends.AutoGain	    0    
DWDR Lim	ParamEx.[0].BroadTrends.Gain		    50  
AE Sensitivity  Param.[0].AeSensitivity			    1  
Defogging	Param.[0].ClearFog.Enable		        False  
Defog Rating	Param.[0].ClearFog.level		    50  
AGC		Param.[0].GainParam.AutoGain		        1  
AGC Limit	Param.[0].GainParam.Gain		        70  
Slow Shutter	Param.[0].EsShutter			        '0x00000000'  
IR_Cut		Param.[0].IRCUTMode			            0  
DayNtLevel	Param.[0].Day_nfLevel			        0  
NightNtLevel	Param.[0].Night_nfLevel			    0  
Mirror		Param.[0].PictureMirror			        '0x00000000'  
Flip		Param.[0].PictureFlip			        '0x00000000'  
Anti Flicker	Param.[0].RejectFlicker			    '0x00000000'  
IR Swap		Param.[0].IrcutSwap			            0  
```

Video Encoding Dialog Settings  
==============================  
Main Channel, Extra Channel identical except ExtraFormat  
------------  
```bash
Compression	Simplify.Encode.[0].MainFormat.Compression		        'H.264'  
Resolution	Simplify.Encode.[0].MainFormat.Video.Resolution		    '720P'  
FPS		Simplify.Encode.[0].MainFormat.Video.FPS		            25  
Bit Rate Type	Simplify.Encode.[0].MainFormat.Video.BitRateControl	'VBR'  
Quality		Simplify.Encode.[0].MainFormat.Video.Quality		    6  
BitRate		Simplify.Encode.[0].MainFormat.Video.BitRate		    4096  
I Frame Int	Simplify.Encode.[0].MainFormat.Video.GOP		        2  
Video on	Simplify.Encode.[0].MainFormat.VideoEnable		        True  
Audio On	Simplify.Encode.[0].MainFormat.AudioEnable		        False  
```
Location of other parameters unknown  

Color Dialog Settings
=====================  
```bash
brightness    AVEnc.VideoColor.[0].["VideoColorParam"]["Brightness"]  
contrast      AVEnc.VideoColor.[0].["VideoColorParam"]["Contrast"]   
saturation    AVEnc.VideoColor.[0].["VideoColorParam"]["Saturation"]   
hue           AVEnc.VideoColor.[0].["VideoColorParam"]["Hue"]   
gain          AVEnc.VideoColor.[0].["VideoColorParam"]["Gain"]  
horizontal    AVEnc.VideoColor.[0].["VideoColorParam"]["Acutance"]  
```

Network Dialog Settings  
=======================  
``` bash
Net Card	NetWork.NetCommon.DeviceType		1  
DHCP Enable	NetWork.NetDHCP.[0].Enable		    False  
IP Address	NetWork.NetCommon.HostIP		    '0x0A01A8C0'  
Subnet		NetWork.NetCommon.Submask		    '0xFFFFFFFF'  
Gateway		NetWork.NetCommon.GateWay		    '0x0A01A8C0'  
Primary DNS	NetWork.NetDNS.Address			    '0xFE01A8C0'  
Secondary DNS	NetWork.NetDNS.SpareAddress		'0x00000000'  
Media Port	NetWork.NetCommon.TCPPort		    34567  
HTTP Port	NetWork.NetCommon.HttpPort		    80  
Onvif Port	unknown
Device Info	NetWork.NetCommon.MAC			    '00:12:31:00:00:00'  
High Speed	NetWork.NetCommon.UseHSDownLoad		True  
Transfer Policy	NetWork.NetCommon.TransferPlan	'Fluency'  
Product ID	NetWork.NetCommon.HostName		    'blank'  
```  
General Dialog  
=============  
``` bash
TimeZone	Read from NTP settings  
System Time	call get_time()  
Date Format	General.Location.DateFormat		    'YYMMDD'  
DST		General.Location.DSTRule		        'Off'  
Date Separator	General.Location.DateSeparator	'-'  
Time Format	General.Location.TimeFormat		    '24'  
Language	General.Location.Language		    'English'  
HDD Full	????  
DVR No		????  
Video Standard	General.Location.VideoFormat	'PAL'  
Auto Logout	General.AutoLogout			        0  
```

AutoMaintain Dialog  
===================  
``` bash
Auto Reboot 	General.AutoMaintain.AutoRebootDay		'Tuesday'  
At		General.AutoMaintain.AutoRebootHour		        2  
Auto Delete	General.AutoMaintain.AutoDeleteFilesDays	0  
Auto Upgrade	???  
```
Netservice NTP settings Dialog  
==============================  
``` bash
Enable		NetWork.NetNTP.Enable				True  
Custom		NetWork.NetNTP.TimeZone				13  
Autoselect	set the above to 14 for some reason  
Server IP	NetWork.NetNTP.Server.Name			'192.168.1.151'  
Port		NetWork.NetNTP.Server.Port			123  
Upgrade period	NetWork.NetNTP.UpdatePeriod		60  
```

Netservice Cloud settings Dialog  
-------------------------------  
```bash
Enable      NetWork.Nat.NatEnable   True 
MTU         NetWork.Nat.XMeyeMTU    1280
Addr        NetWork.Nat.Addr        secu100.net

```
Connects to secu100.net and push.xmeye.cn every 10 seconds if enabled  
using ports 80 and 8000  but possibly others.  

Info/Version Dialog  
===================  
``` bash
Record Chan  
Extra Chan  
Alarm In  
Alarm Out  
System  
Build Date  
System Status  
Serial ID  
Nat Status  
Nat Stat Code  
```