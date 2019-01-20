#include <DHT.h>  // Including library for dht

#include <ESP8266WiFi.h>
 
String apikey = "3GH0UPMWYROH8UQ9";     //  Enter your Write API key from ThingSpeak

const char *ssid =  "moto x4 4055";     // replace with your wifi ssid and wpa2 key
const char *password =  "9876531578";
const char* server = "api.thingspeak.com";

DHT d(D0, DHT11);
WiFiClient client;

void setup() {

  Serial.begin(115200);
       delay(10);
       d.begin();
 
       Serial.println("Connecting to ");
       Serial.println(ssid);
       WiFi.begin(ssid, password);
 
      while (WiFi.status() != WL_CONNECTED) 
     {
            delay(500);
            Serial.print(".");
     }
      Serial.println("");
      Serial.println("WiFi connected");
}
//www.thingspeak.com
void loop() {
int t=d.readTemperature();
int h=d.readHumidity();

if(client.connect(server,80)){
  //GET https://api.thingspeak.com/update?api_key=3GH0UPMWYROH8UQ9&field1=0

String getstr = apikey;
getstr += "&field1=";
getstr += String(t);
getstr += "&field2=";
getstr += String(h);
getstr += "\r\n\r\n";

client.print("GET /update HTTP/1.1\n");
client.print("Host: api.thingspeak.com\n");
client.print("Connection: close\n");
client.print("X-THINGSPEAKAPIKEY: "+apikey+"\n");
client.print("Content-Type: application/x-www-form-urlencoded\n");

client.print("Content-Length: ");
client.print(getstr.length());
client.print("\n\n");
client.print(getstr);

Serial.print("Temperature: ");
Serial.println(t);
Serial.print("Humidity: ");
Serial.print(h);
}
client.stop();
delay(15000);
}
