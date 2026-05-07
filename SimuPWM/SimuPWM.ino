#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

const char* WIFI_SSID = "Lopes";
const char* WIFI_PASS = "12345678";

WiFiUDP udp;
const int UDP_PORT = 5005;

const float v_max = 0.47f;

float v_cmd = 0.0f, d_cmd = 0.0f;

unsigned long lastPacketTime = 0;
const unsigned long timeoutMs = 300;

int AIN1 = D1;
int AIN2 = D2;
int STBY = D3;
int BIN1 = D4;
int BIN2 = D5;
int PWMA = D0;
int PWMB = D6;

bool aux = 0;

// Simulação PWM direção
int pwm_d_sim = 0;
int step_pwm_d = 5;
unsigned long lastPwmUpdate = 0;
const unsigned long pwmInterval = 20; // ms

void connectWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.print("A ligar ao WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi ligado");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

bool parseCommand(const char* msg, float& v, float& d) {
    return sscanf(msg, "V:%f D:%f", &v, &d) == 2;
}

float grausParaRadianos(float graus) {
    return graus * PI / 180.0f;
}

const float delta_max = grausParaRadianos(30.0f);

int velocity_to_pwm(float v) {
    float ratio = constrain(fabs(v) / v_max, 0.0f, 1.0f);
    return (int)(ratio * 255.0f);
}

int delta_to_pwm(float delta) {
    float ratio = constrain(fabs(delta) / delta_max, 0.0f, 1.0f);
    return (int)(ratio * 255.0f);
}

void setup() {
    Serial.begin(115200);

    pinMode(PWMA, OUTPUT);
    pinMode(AIN1, OUTPUT);
    pinMode(AIN2, OUTPUT);
    pinMode(STBY, OUTPUT);
    pinMode(BIN1, OUTPUT);
    pinMode(BIN2, OUTPUT);
    pinMode(PWMB, OUTPUT);

    digitalWrite(STBY, HIGH);

    analogWrite(PWMA, 0);
    analogWrite(PWMB, 0);

    connectWifi();

    if (udp.begin(UDP_PORT)) {
        Serial.print("A escutar UDP na porta ");
        Serial.println(UDP_PORT);
    } else {
        Serial.println("Erro a iniciar UDP");
    }
}

void loop() {
    int packetSize = udp.parsePacket();

    if (packetSize) {
        char incoming[128];
        int len = udp.read(incoming, sizeof(incoming) - 1);

        if (len > 0) {
            incoming[len] = '\0';
            lastPacketTime = millis();

            Serial.print("Recebido: ");
            Serial.println(incoming);

            float v, d;
            if (parseCommand(incoming, v, d)) {
                v_cmd = v;
                d_cmd = d;
            } else {
                Serial.println("Formato invalido");
            }
        }
    }

    bool timeout = (millis() - lastPacketTime) > timeoutMs;

    int pwm_v = velocity_to_pwm(v_cmd);

    // -----------------------------
    // SIMULAÇÃO PROGRESSIVA pwm_d
    // -----------------------------
    if (millis() - lastPwmUpdate >= pwmInterval) {
        lastPwmUpdate = millis();

        pwm_d_sim += step_pwm_d;

        if (pwm_d_sim >= 255) {
            pwm_d_sim = 255;
            step_pwm_d = -step_pwm_d;
        }

        if (pwm_d_sim <= 0) {
            pwm_d_sim = 0;
            step_pwm_d = -step_pwm_d;
        }

        Serial.print("PWM direção simulado: ");
        Serial.println(pwm_d_sim);
    }

    int pwm_d = pwm_d_sim;

    // -----------------------------
    // MOTOR VELOCIDADE
    // -----------------------------
    if (v_cmd > 0) {
        digitalWrite(STBY, HIGH);
        digitalWrite(BIN1, LOW);
        digitalWrite(BIN2, HIGH);
        analogWrite(PWMB, pwm_v);
        aux = 1;
    }
    else if (v_cmd < 0) {
        digitalWrite(STBY, HIGH);
        digitalWrite(BIN1, HIGH);
        digitalWrite(BIN2, LOW);
        analogWrite(PWMB, pwm_v);
    }
    else {
        analogWrite(PWMB, 0);
    }

    // -----------------------------
    // MOTOR DIREÇÃO COM PWM SIMULADO
    // -----------------------------
    if (step_pwm_d > 0) {
        digitalWrite(STBY, HIGH);
        digitalWrite(AIN1, LOW);
        digitalWrite(AIN2, HIGH);
        analogWrite(PWMA, pwm_d);
    }
    else if (step_pwm_d < 0) {
        digitalWrite(STBY, HIGH);
        digitalWrite(AIN1, HIGH);
        digitalWrite(AIN2, LOW);
        analogWrite(PWMA, pwm_d);
    }
    else {
        analogWrite(PWMA, 0);
    }
}