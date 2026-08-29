/**
 * esp32_tinyml_anomaly.ino
 *
 * Real ESP32 firmware for on-device vibration anomaly detection: reads an
 * MPU6050 accelerometer, buffers a 64-sample window, runs it through the
 * embedded quantized autoencoder (model_data.h, produced by
 * model/export_c_array.py), computes the reconstruction error, and
 * publishes an alert over MQTT if it exceeds the trained threshold -
 * exactly the same topic scheme as simulator/virtual_sensor.py, so the
 * dashboard doesn't know or care which one is running.
 *
 * IMPORTANT - NOT COMPILE-TESTED, same limitation as the other two ESP32
 * projects in this series: this sandbox can't reach the Arduino/Espressif
 * toolchain. Written carefully against the standard APIs below, but you
 * are its first compiler - treat "flash and verify" as mandatory.
 *
 * Required libraries (Arduino Library Manager):
 *   - Adafruit MPU6050 + Adafruit Unified Sensor + Adafruit BusIO
 *   - PubSubClient (Nick O'Leary)
 *   - TensorFlow Lite for Microcontrollers - this project was written
 *     against the community-maintained `esp-tflite-micro` for the ESP32
 *     platform (github.com/espressif/esp-tflite-micro), which tracks the
 *     upstream TFLM API used below (tflite::MicroInterpreter etc.). If
 *     you use a different TFLM port, the includes/namespaces may need
 *     small adjustments - check that library's example sketch.
 *
 * Wiring:
 *   MPU6050 SDA -> GPIO21, SCL -> GPIO22 (default ESP32 I2C pins)
 *   Alert LED (optional) -> GPIO2 (many ESP32 dev boards have one built in)
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <PubSubClient.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "config.h"          // WIFI_SSID, WIFI_PASSWORD, MQTT_HOST, MQTT_PORT, DEVICE_ID
#include "model_data.h"       // g_model_data[], g_model_data_len - from model/export_c_array.py
#include "model_params.h"     // kNormalizationMean/Std, kWindowSize, kAnomalyThreshold

#define ALERT_LED_PIN 2
#define WINDOW_INTERVAL_MS 1000UL
#define TENSOR_ARENA_SIZE (8 * 1024)  // 8KB is comfortable headroom for a model this small

Adafruit_MPU6050 mpu;
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

String topicStatus       = String("machine/") + DEVICE_ID + "/status";
String topicRms          = String("machine/") + DEVICE_ID + "/vibration/rms";
String topicAnomalyScore = String("machine/") + DEVICE_ID + "/vibration/anomaly_score";
String topicAlert        = String("machine/") + DEVICE_ID + "/vibration/alert";

// ---------------------------------------------------------------------
// TensorFlow Lite Micro setup
// ---------------------------------------------------------------------
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Only the 4 ops this small dense autoencoder actually uses - keeps the
// resolver (and therefore flash usage) minimal versus AllOpsResolver.
tflite::MicroMutableOpResolver<2> resolver;

float windowBuffer[kWindowSize];
int windowIndex = 0;
bool lastAlertState = false;
unsigned long lastWindowMs = 0;

// ---------------------------------------------------------------------
void setupModel() {
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch!");
        while (true) { delay(1000); }
    }

    resolver.AddFullyConnected();
    resolver.AddRelu();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed - increase TENSOR_ARENA_SIZE");
        while (true) { delay(1000); }
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

/** Runs one window through the model, returns the reconstruction MSE in
 * normalized (float) space - mirrors simulator/anomaly_detector.py. */
float scoreWindow(const float *rawWindow) {
    float inScale = input->params.scale;
    int inZero = input->params.zero_point;
    float outScale = output->params.scale;
    int outZero = output->params.zero_point;

    float normed[64];
    for (int i = 0; i < kWindowSize; i++) {
        normed[i] = (rawWindow[i] - kNormalizationMean) / kNormalizationStd;
        int8_t q = (int8_t)round(normed[i] / inScale + inZero);
        input->data.int8[i] = q;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke() failed");
        return 0.0f;
    }

    float mse = 0.0f;
    for (int i = 0; i < kWindowSize; i++) {
        float recon = (output->data.int8[i] - outZero) * outScale;
        float diff = normed[i] - recon;
        mse += diff * diff;
    }
    return mse / kWindowSize;
}

// ---------------------------------------------------------------------
void connectWiFi() {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
}

void connectMqtt() {
    mqtt.setServer(MQTT_HOST, MQTT_PORT);
    while (!mqtt.connected()) {
        Serial.print("Connecting to MQTT...");
        if (mqtt.connect(DEVICE_ID, topicStatus.c_str(), 0, true, "offline")) {
            Serial.println("connected");
            mqtt.publish(topicStatus.c_str(), "online", true);
        } else {
            Serial.print("failed, rc=");
            Serial.println(mqtt.state());
            delay(2000);
        }
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(ALERT_LED_PIN, OUTPUT);

    Wire.begin();
    if (!mpu.begin()) {
        Serial.println("MPU6050 not found - check wiring");
        while (true) { delay(1000); }
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);

    setupModel();
    connectWiFi();
    connectMqtt();
}

void loop() {
    if (!mqtt.connected()) {
        connectMqtt();
    }
    mqtt.loop();

    if (millis() - lastWindowMs < (WINDOW_INTERVAL_MS / kWindowSize)) {
        return;  // pace sample collection so a full window spans WINDOW_INTERVAL_MS
    }
    lastWindowMs = millis();

    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    windowBuffer[windowIndex++] = accel.acceleration.x;  // single axis, matching the training data model

    if (windowIndex < kWindowSize) {
        return;  // keep buffering until we have a full window
    }
    windowIndex = 0;

    float rms = 0.0f;
    for (int i = 0; i < kWindowSize; i++) rms += windowBuffer[i] * windowBuffer[i];
    rms = sqrt(rms / kWindowSize);

    float anomalyScore = scoreWindow(windowBuffer);
    bool isAnomaly = anomalyScore > kAnomalyThreshold;

    mqtt.publish(topicRms.c_str(), String(rms, 4).c_str());
    mqtt.publish(topicAnomalyScore.c_str(), String(anomalyScore, 6).c_str());

    if (isAnomaly != lastAlertState) {
        mqtt.publish(topicAlert.c_str(), isAnomaly ? "1" : "0", true);
        lastAlertState = isAnomaly;
    }
    digitalWrite(ALERT_LED_PIN, isAnomaly ? HIGH : LOW);
}
