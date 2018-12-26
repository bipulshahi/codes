import serial
import urllib2
import time
ser=serial.Serial('COM3',9600,timeout=5)

def main():
      print 'starting...'
      URL='https://api.thingspeak.com/update?api_key=6HF03S7QI8JF68W5'
      print 'wait......'

      while True:
            t=ser.read(2)
            h=ser.readline()
            finalURL=URL+"&field1=%s&field2=%s"%(t,h)
            print finalURL
            s=urllib2.urlopen(finalURL);
            print t + " " + h + " "
            s.close()
            time.sleep(30)

main()
