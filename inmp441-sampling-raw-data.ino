// ESP32-S3 + INMP441 @16 kHz, raw int16 lines for Edge Impulse DF

#include <Arduino.h>
#include "driver/i2s.h"

// ===== Pins (adjust to your wiring) =====
#define I2S_PORT        I2S_NUM_0
#define I2S_BCLK_PIN    12   // SCK/BCLK
#define I2S_WS_PIN      11   // LRCL/WS
#define I2S_SD_PIN      13   // SD/DOUT  (INMP441 DOUT -> ESP32 data_in)

// ===== Audio params =====
#define SAMPLE_RATE     16000
#define SERIAL_BAUD     2000000

// Read in blocks for efficiency
#define BLOCK_SAMPLES   256      // samples per read
static int32_t i2s_block[BLOCK_SAMPLES];  // raw 32-bit I2S words

static inline int16_t s24_to_s16(int32_t w) {
  // INMP441: 24-bit signed left-justified in 32-bit word (MSB-aligned).
  // Move to bits [23:0], then sign-extend to 32, then downscale to 16.
  int32_t s24 = w >> 8;                  // now top 24 bits in [31:8]
  s24 = (s24 & 0x00FFFFFF);              // keep 24 bits
  if (s24 & 0x00800000) s24 |= 0xFF000000; // sign-extend from bit 23
  return (int16_t)(s24 >> 8);            // 24->16 with gentle downscale
}

static void i2s_install_and_start() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,  // 24-bit in 32-bit slot
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // INMP441 L/R pin = GND -> Left
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BLOCK_SAMPLES,
    .use_apll = true,       // better clock accuracy at 16 kHz
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
  ESP_ERROR_CHECK(i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO));

  // Clear any stale DMA contents and realign to frame boundary
  i2s_zero_dma_buffer(I2S_PORT);

  // Throw away ~150 ms to let mic bias & DMA settle
  const uint32_t throw_samples = (SAMPLE_RATE * 150) / 1000; // ~2400
  uint32_t tossed = 0;
  size_t br;
  while (tossed < throw_samples) {
    i2s_read(I2S_PORT, (void*)i2s_block, sizeof(i2s_block), &br, portMAX_DELAY);
    tossed += (br / sizeof(int32_t));
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  // IMPORTANT: Do not print any non-numeric text after this point.
  i2s_install_and_start();
}

void loop() {
  size_t bytes_read = 0;
  // Read a whole block of 32-bit words
  if (i2s_read(I2S_PORT, (void*)i2s_block, sizeof(i2s_block), &bytes_read, portMAX_DELAY) == ESP_OK) {
    int n = bytes_read / sizeof(int32_t);
    for (int i = 0; i < n; ++i) {
      int16_t s = s24_to_s16(i2s_block[i]);
      Serial.println(s);  // one sample per line
    }
  }
}

