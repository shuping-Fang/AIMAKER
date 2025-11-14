/**
 * ESP32-S3 + INMP441 + Edge Impulse (float32) – Continuous KWS
 * Model window ≈700 ms (11520 samples @16 kHz)
 * Pre-processing: High-pass + Pre-emphasis + AGC
 * Hop ≈ 100 ms (WIN/7)
 * Commands: light_on / light_off / unknown  →  Relay + LED
 */

#include <Arduino.h>
#include <audiotest2_inferencing.h>   // <-- your exported Edge Impulse model
#include "driver/i2s.h"
#include <cstring>

// ===== I²S pins (ESP32-S3) =====
#define I2S_PORT        I2S_NUM_0
#define I2S_BCLK_PIN    12
#define I2S_WS_PIN      11
#define I2S_SD_PIN      13
#define I2S_CHANNEL_FMT I2S_CHANNEL_FMT_ONLY_LEFT   // L/R → GND = Left

// ===== Relay / LED pins =====
#define RELAY1_PIN      4
#define RELAY2_PIN      5
#define RELAY3_PIN      6
#define RELAY4_PIN      7
#define LED_PIN         18
#define RELAY_ACTIVE_LEVEL  LOW
#define RELAY_IDLE_LEVEL    (RELAY_ACTIVE_LEVEL == LOW ? HIGH : LOW)

// ===== Inference parameters =====
static_assert(EI_CLASSIFIER_FREQUENCY == 16000, "Model must be 16 kHz");
const size_t WIN_SAMPLES = EI_CLASSIFIER_RAW_SAMPLE_COUNT;   // 11520
const size_t HOP_SAMPLES = WIN_SAMPLES / 7;                  // ≈100 ms
const float  MIN_CONF = 0.60f;
const float  MARGIN_OVER_UNKNOWN = 0.15f;
const int    VOTE_SIZE = 7;
const int    VOTE_NEED = 4;
const uint32_t REFRACT_MS = 700;

// ===== Pre-processing filters =====
#define HPF_ALPHA  0.995f
#define PRE_EMPH_A 0.97f

// ===== VAD / AGC settings =====
const float VAD_RMS_SILENCE = 0.015f;
const float AGC_TARGET_RMS  = 0.05f;
const float AGC_MAX_GAIN    = 20.0f;

// ===== Buffers / state =====
static int16_t pcm_i16[WIN_SAMPLES];
static float   pcm_f32[WIN_SAMPLES];
static int16_t hop_buf[HOP_SAMPLES];
static int8_t  votes[VOTE_SIZE];
static int     vote_idx = 0;
static uint32_t last_trigger_ms = 0;

// ===== Label indices =====
static int8_t LABEL_LIGHT_ON  = -1;
static int8_t LABEL_LIGHT_OFF = -1;
static int8_t LABEL_UNKNOWN   = -1;

// ---------- I²S setup ----------
void i2s_install_and_start() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = EI_CLASSIFIER_FREQUENCY,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT,
    .communication_format =
        (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 6,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num  = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD_PIN
  };
  ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT, &cfg, 0, NULL));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT, &pins));
  ESP_ERROR_CHECK(i2s_set_clk(I2S_PORT, EI_CLASSIFIER_FREQUENCY,
                              I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO));
}

// ---------- I²S helpers ----------
inline int16_t i2s32_to_i16(int32_t s32) {
  uint32_t u = (uint32_t)s32 >> 8;
  int32_t s24 = (u & 0x00800000) ? (int32_t)(u | 0xFF000000) : (int32_t)u;
  return (int16_t)(s24 >> 8);
}
size_t i2s_read_i16(int16_t *dst, size_t n) {
  size_t total = 0;
  while (total < n) {
    int32_t s32; size_t br = 0;
    if (i2s_read(I2S_PORT, &s32, sizeof(s32), &br, portMAX_DELAY) != ESP_OK)
      break;
    if (br != sizeof(s32)) break;
    dst[total++] = i2s32_to_i16(s32);
  }
  return total;
}

// ---------- Filters / AGC ----------
void highpass_inplace(float *x, size_t n, float a) {
  if (!n) return;
  float y_prev = 0, x_prev = x[0];
  for (size_t i = 0; i < n; i++) {
    float y = a * (y_prev + x[i] - x_prev);
    x_prev = x[i]; x[i] = y; y_prev = y;
  }
}
void preemphasis_inplace(float *x, size_t n, float a) {
  if (!n) return;
  float prev = x[0];
  for (size_t i = n - 1; i > 0; i--) x[i] = x[i] - a * x[i - 1];
  x[0] = x[0] - a * prev;
}
float compute_rms(const float *x, size_t n) {
  double acc = 0.0; for (size_t i=0;i<n;i++) acc += (double)x[i]*x[i];
  return sqrt(acc / (double)n);
}
void agc_normalize(float *x, size_t n, float target, float max_gain) {
  float r = compute_rms(x, n);
  if (r <= 1e-7f) return;
  float g = target / r;
  if (g > max_gain) g = max_gain;
  for (size_t i=0;i<n;i++) {
    float v = x[i]*g;
    if (v>0.98f) v=0.98f; if (v<-0.98f) v=-0.98f;
    x[i]=v;
  }
}

// ---------- EMA smoother ----------
struct Ema {
  bool init=false;
  float y[EI_CLASSIFIER_LABEL_COUNT];
  void reset(){init=false;}
  void step(const ei_impulse_result_t& r,float a=0.7f){
    if(!init){for(size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++)y[i]=r.classification[i].value;init=true;return;}
    for(size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++)
      y[i]=a*y[i]+(1-a)*r.classification[i].value;
  }
} ema;

// ---------- Utils ----------
void to_float_norm(const int16_t *src,float *dst,size_t n){
  for(size_t i=0;i<n;i++) dst[i]=(float)src[i]/32768.0f;
}
int top1_label(const float *scores,float &best_score){   // ✅ fixed bug here
  best_score=scores[0]; int best=0;
  for(size_t i=1;i<EI_CLASSIFIER_LABEL_COUNT;i++){
    if(scores[i]>best_score){best_score=scores[i]; best=(int)i;}  // correct
  }
  return best;
}
bool vote_and_decide(int label,int &winner){
  votes[vote_idx]=label; vote_idx=(vote_idx+1)%VOTE_SIZE;
  int c[EI_CLASSIFIER_LABEL_COUNT]={0};
  for(int i=0;i<VOTE_SIZE;i++){int v=votes[i]; if(v>=0&&v<EI_CLASSIFIER_LABEL_COUNT)c[v]++;}
  int best=-1,bestc=-1;
  for(int i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++) if(c[i]>bestc){best=i;bestc=c[i];}
  winner=best; return bestc>=VOTE_NEED;
}
void apply_action_from_label(int label,float score){
  uint32_t now=millis(); if(now-last_trigger_ms<REFRACT_MS)return;
  if(label==LABEL_LIGHT_ON&&score>=MIN_CONF){
    digitalWrite(RELAY1_PIN,RELAY_ACTIVE_LEVEL); digitalWrite(LED_PIN,HIGH);
    last_trigger_ms=now; Serial.printf("[ACTION] light_on (%.2f)\n",score);
  }else if(label==LABEL_LIGHT_OFF&&score>=MIN_CONF){
    digitalWrite(RELAY1_PIN,RELAY_IDLE_LEVEL); digitalWrite(LED_PIN,LOW);
    last_trigger_ms=now; Serial.printf("[ACTION] light_off (%.2f)\n",score);
  }
}

// ---------- Arduino ----------
void setup(){
  Serial.begin(115200); while(!Serial){} delay(200);
  Serial.println("\n[EI] ESP32-S3 continuous KWS (final fixed)");
  Serial.printf("RAW=%u HOP=%u FREQ=%u\n",
    (unsigned)EI_CLASSIFIER_RAW_SAMPLE_COUNT,(unsigned)HOP_SAMPLES,(unsigned)EI_CLASSIFIER_FREQUENCY);

  pinMode(RELAY1_PIN,OUTPUT); pinMode(RELAY2_PIN,OUTPUT);
  pinMode(RELAY3_PIN,OUTPUT); pinMode(RELAY4_PIN,OUTPUT); pinMode(LED_PIN,OUTPUT);
  digitalWrite(RELAY1_PIN,RELAY_IDLE_LEVEL); digitalWrite(RELAY2_PIN,RELAY_IDLE_LEVEL);
  digitalWrite(RELAY3_PIN,RELAY_IDLE_LEVEL); digitalWrite(RELAY4_PIN,RELAY_IDLE_LEVEL);
  digitalWrite(LED_PIN,LOW);
  i2s_install_and_start();

  // Map labels
  for(size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++){
    const char* n=ei_classifier_inferencing_categories[i];
    if(strcmp(n,"light_on")==0||strcmp(n,"light on")==0) LABEL_LIGHT_ON=i;
    if(strcmp(n,"light_off")==0||strcmp(n,"light off")==0) LABEL_LIGHT_OFF=i;
    if(strcmp(n,"unknown")==0||strcmp(n,"noise")==0||strstr(n,"unk")) LABEL_UNKNOWN=i;
  }
  Serial.printf("Labels → on:%d off:%d unk:%d\n",LABEL_LIGHT_ON,LABEL_LIGHT_OFF,LABEL_UNKNOWN);

  size_t got=i2s_read_i16(pcm_i16,WIN_SAMPLES);
  if(got!=WIN_SAMPLES) Serial.printf("I2S underrun %u/%u\n",(unsigned)got,(unsigned)WIN_SAMPLES);
  memset(votes,-1,sizeof(votes)); ema.reset();
  Serial.println("Say ‘light on’ / ‘light off’");
}

void loop(){
  to_float_norm(pcm_i16,pcm_f32,WIN_SAMPLES);
  highpass_inplace(pcm_f32,WIN_SAMPLES,HPF_ALPHA);
  preemphasis_inplace(pcm_f32,WIN_SAMPLES,PRE_EMPH_A);
  agc_normalize(pcm_f32,WIN_SAMPLES,AGC_TARGET_RMS,AGC_MAX_GAIN);

  float rms=compute_rms(pcm_f32,WIN_SAMPLES);
  if(rms<VAD_RMS_SILENCE){
    Serial.printf("[VAD] silence rms=%.4f\n",rms);
  }else{
    signal_t signal;
    if(numpy::signal_from_buffer(pcm_f32,WIN_SAMPLES,&signal)!=0)return;
    ei_impulse_result_t res={0};
    if(run_classifier(&signal,&res,false)!=EI_IMPULSE_OK)return;

    ema.step(res,0.7f);
    float scores[EI_CLASSIFIER_LABEL_COUNT];
    for(size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++)
      scores[i]=ema.init?ema.y[i]:res.classification[i].value;

    float best_score; int best=top1_label(scores,best_score);
    float unk=(LABEL_UNKNOWN>=0?scores[LABEL_UNKNOWN]:0.0f);
    float margin=best_score-unk;

    Serial.printf("[rms=%.3f] ",rms);
    for(size_t i=0;i<EI_CLASSIFIER_LABEL_COUNT;i++)
      Serial.printf("%s:%.2f ",ei_classifier_inferencing_categories[i],scores[i]);
    Serial.printf("| top=%s(%.2f) margin=%.2f\n",
      ei_classifier_inferencing_categories[best],best_score,margin);

    bool pass=(best_score>=MIN_CONF)&&(margin>=MARGIN_OVER_UNKNOWN);
    if(pass){
      int winner;
      if(vote_and_decide(best,winner))
        apply_action_from_label(winner,scores[winner]);
    }
  }

  // Slide window
  memmove(pcm_i16,pcm_i16+HOP_SAMPLES,(WIN_SAMPLES-HOP_SAMPLES)*sizeof(int16_t));
  size_t got=i2s_read_i16(hop_buf,HOP_SAMPLES);
  if(got!=HOP_SAMPLES) Serial.printf("I2S hop underrun %u/%u\n",(unsigned)got,(unsigned)HOP_SAMPLES);
  memcpy(pcm_i16+(WIN_SAMPLES-HOP_SAMPLES),hop_buf,HOP_SAMPLES*sizeof(int16_t));
}
