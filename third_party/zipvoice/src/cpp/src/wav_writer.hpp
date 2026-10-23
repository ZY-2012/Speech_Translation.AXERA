/**************************************************************************************************
 * ZipVoice AXERA C++ Port
 *
 * Simple WAV file writer for float32 mono audio.
 **************************************************************************************************/

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>

class WavWriter {
public:
    /**
     * Write float32 mono audio samples to a WAV file.
     *
     * @param path        Output file path
     * @param samples     Float32 audio samples in range [-1.0, 1.0]
     * @param sample_rate Sample rate in Hz (e.g., 24000)
     * @return true on success
     */
    static bool Write(const std::string& path,
                      const std::vector<float>& samples,
                      int sample_rate) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        int num_samples = static_cast<int>(samples.size());
        int byte_rate = sample_rate * 2;  // 16-bit mono
        int data_size = num_samples * 2;
        int chunk_size = 36 + data_size;

        // RIFF header
        Write4CC(file, "RIFF");
        WriteU32LE(file, chunk_size);
        Write4CC(file, "WAVE");

        // fmt chunk
        Write4CC(file, "fmt ");
        WriteU32LE(file, 16);           // chunk size
        WriteU16LE(file, 1);            // PCM format
        WriteU16LE(file, 1);            // mono
        WriteU32LE(file, sample_rate);
        WriteU32LE(file, byte_rate);
        WriteU16LE(file, 2);            // block align
        WriteU16LE(file, 16);           // bits per sample

        // data chunk
        Write4CC(file, "data");
        WriteU32LE(file, data_size);

        // Convert float32 to int16 (with clipping)
        for (float sample : samples) {
            // Clamp to [-1, 1]
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
            int16_t val = static_cast<int16_t>(sample * 32767.0f);
            WriteU16LE(file, static_cast<uint16_t>(val));
        }

        file.close();
        return true;
    }

private:
    static void WriteU16LE(std::ofstream& f, uint16_t v) {
        f.put(v & 0xFF);
        f.put((v >> 8) & 0xFF);
    }

    static void WriteU32LE(std::ofstream& f, uint32_t v) {
        f.put(v & 0xFF);
        f.put((v >> 8) & 0xFF);
        f.put((v >> 16) & 0xFF);
        f.put((v >> 24) & 0xFF);
    }

    static void Write4CC(std::ofstream& f, const char* s) {
        f.write(s, 4);
    }
};
