// qwen-tts.cpp: thin CLI wrapper around the qwentts.cpp public ABI.
// Parses arguments, reads the optional reference WAV plus transcript,
// hands off to qt_synthesize and writes the resulting waveform as a
// WAV file. All synthesis logic, mode validation and seed resolution
// live behind the qwen_* facade declared in qwen.h.
//
// Talker variants: 0.6B-Base / 0.6B-CustomVoice / 1.7B-Base /
// 1.7B-CustomVoice / 1.7B-VoiceDesign. The decoder path is selected
// from GGUF metadata at qt_init time. The CLI surface mirrors the
// omnivoice.cpp tooling: kebab-case flags, --format wav16/wav24/wav32,
// -o '-' streams to stdout, --seed -1 means non deterministic
// (resolved inside qt_synthesize), the utterance text is read from stdin.

#include "audio-io.h"
#include "qwen.h"
#include "pipeline-codec.h"
#include "pipeline-tts.h"
#include "qwen-internal.h"
#include "speaker-encoder-extract.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", qt_version());
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n"
            "  -o <path>               Output WAV (24 kHz mono). '-' streams to stdout (pipe friendly).\n\n"
            "Input:\n"
            "  stdin                   Target text to synthesise. Read fully then synthesised in one shot.\n\n"
            "Optional:\n"
            "  --format <fmt>          WAV output format: wav16, wav24, wav32 (default: wav16)\n"
            "  --lang <name>           Language label (default: english)\n"
            "  --instruct <str>        Style instruction. Required for VoiceDesign, optional for\n"
            "                          CustomVoice, rejected for Base\n"
            "  --speaker <name>        Speaker name (CustomVoice only)\n"
            "  --ref-wav <path>        Reference WAV for voice cloning (Base only)\n"
            "  --ref-text <path>       Transcript file for the reference (enables ICL clone mode)\n"
            "  --max-new <n>           Max new audio frames (default: 2048)\n"
            "  --codec-chunk-dur <f>   Codec decode chunk duration in seconds (default: 24.0)\n"
            "  --codec-left-dur <f>    Codec decode left context duration in seconds (default: 2.0)\n\n"
            "Sampling:\n"
            "  --seed <int>            Sampling seed (default: -1 for random)\n"
            "  --greedy                Disable stochastic sampling on both stacks\n"
            "  --temp <f>              Talker temperature (default: 0.9)\n"
            "  --top-k <n>             Talker top-k (default: 50, 0 disables)\n"
            "  --top-p <f>             Talker top-p (default: 1.0)\n"
            "  --rep-pen <f>           Talker repetition penalty (default: 1.05)\n"
            "  --sub-temp <f>          Sub-talker temperature (default: 0.9)\n"
            "  --sub-top-k <n>         Sub-talker top-k (default: 50)\n"
            "  --sub-top-p <f>         Sub-talker top-p (default: 1.0)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states + V to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors (f32) to <dir>\n\n"
            "Daemon (--loop):\n"
            "  --loop                  Keep model loaded; read requests via stdin loop.\n"
            "                          Binary protocol: [4B LE json_len][json][4B LE status]\n"
            "                          [4B LE meta_len][meta][streaming chunks][4B LE 0].\n",
            prog);
}

struct Args {
    const char * model;
    const char * codec;
    const char * lang;
    const char * instruct;
    const char * speaker;
    const char * ref_wav;
    const char * ref_text_path;
    const char * dump_dir;
    const char * out_wav;
    const char * format;
    int          max_new_tokens;
    int64_t      seed;
    bool         do_sample;
    float        temperature;
    int          top_k;
    float        top_p;
    float        repetition_penalty;
    int          subtalker_top_k;
    float        subtalker_top_p;
    float        subtalker_temperature;
    bool         subtalker_do_sample;
    bool         use_fa;
    bool         clamp_fp16;
    float        codec_chunk_sec;
    float        codec_left_context_sec;
    bool         loop;
};

// Read all of stdin into a string. Trims trailing newlines so a piped
// text file behaves like clean utterance input.
static std::string read_stdin_text() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// Read a small text file into a string. Trims trailing newlines.
static bool read_text_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[CLI] FATAL: cannot open '%s'\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: ftell failed on '%s'\n", path);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        fprintf(stderr, "[CLI] FATAL: fread failed on '%s'\n", path);
        return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers for the --loop daemon protocol (flat objects only)
// ---------------------------------------------------------------------------
static std::string json_str(const std::string & s, const std::string & key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = s.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    std::string out;
    bool esc = false;
    for (; pos < s.size(); pos++) {
        if (esc) {
            switch (s[pos]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                default:   out += s[pos]; break;
            }
            esc = false;
        } else if (s[pos] == '\\') {
            esc = true;
        } else if (s[pos] == '"') {
            break;
        } else {
            out += s[pos];
        }
    }
    return out;
}

static int64_t json_int64(const std::string & s, const std::string & key, int64_t def) {
    auto p = s.find("\"" + key + "\":");
    if (p == std::string::npos) return def;
    p += key.size() + 3;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    if (s.substr(p, 4) == "null" || s.substr(p, 4) == "NULL") return def;
    char * end = nullptr;
    auto v = strtoll(s.c_str() + p, &end, 10);
    return (end == s.c_str() + p) ? def : v;
}

static float json_float(const std::string & s, const std::string & key, float def) {
    auto p = s.find("\"" + key + "\":");
    if (p == std::string::npos) return def;
    p += key.size() + 3;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    if (s.substr(p, 4) == "null" || s.substr(p, 4) == "NULL") return def;
    char * end = nullptr;
    auto v = strtof(s.c_str() + p, (char **)&end);
    return (end == s.c_str() + p) ? def : v;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (auto c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Streaming callback for the --loop daemon protocol
// ---------------------------------------------------------------------------
struct StreamCtx {
    FILE * fp;
    bool   ok;
};

static bool stream_chunk_cb(const float * samples, int n_samples, void * user_data) {
    auto * ctx = (StreamCtx *)user_data;
    if (!ctx->ok || n_samples <= 0) return true;

    uint32_t nbytes = (uint32_t)n_samples * 2u;
    if (fwrite(&nbytes, 4, 1, ctx->fp) != 1) {
        ctx->ok = false;
        return false;
    }
    for (int i = 0; i < n_samples; i++) {
        float f = samples[i];
        if (!std::isfinite(f)) f = 0.0f;
        f = (f < -1.0f) ? -1.0f : (f > 1.0f ? 1.0f : f);
        int16_t pcm = (int16_t)(f * 32767.0f);
        if (fwrite(&pcm, 2, 1, ctx->fp) != 1) {
            ctx->ok = false;
            return false;
        }
    }
    fflush(ctx->fp);
    return true;
}

// ---------------------------------------------------------------------------
// --loop daemon: keep model alive across requests, stdin/stdout protocol
// ---------------------------------------------------------------------------
static int run_loop(const Args & a) {
    qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = a.model;
    iparams.codec_path  = a.codec;
    iparams.use_fa      = a.use_fa;
    iparams.clamp_fp16  = a.clamp_fp16;

    qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Loop] ERROR: %s\n", qt_last_error());
        return 1;
    }
    fprintf(stderr, "[Loop] Model loaded, entering request loop\n");

    std::string last_ref_wav;
    std::string last_ref_text;
    int         ref_n_samples = 0;
    std::unique_ptr<float, decltype(&std::free)> ref_buf(nullptr, std::free);
    std::string ref_text_buf;

    // Cached pre-computed voice data (cleared whenever ref_wav changes)
    std::vector<float>    cached_spk_emb;
    std::vector<int32_t>  cached_ref_codes;
    int                   cached_ref_codes_T = 0;

    uint32_t req_len = 0;
    while (fread(&req_len, 4, 1, stdin) == 1) {
        if (req_len == 0 || req_len > 1024 * 1024) {
            fprintf(stderr, "[Loop] Invalid request length %u\n", req_len);
            continue;
        }
        std::string req((size_t)req_len, '\0');
        if (fread(&req[0], 1, req_len, stdin) != req_len) {
            fprintf(stderr, "[Loop] Short read, exiting\n");
            break;
        }

        std::string req_id      = json_str(req, "id");
        std::string text        = json_str(req, "text");
        std::string ref_wav     = json_str(req, "ref_wav");
        std::string ref_txt_p   = json_str(req, "ref_text");
        std::string lang        = json_str(req, "lang");
        std::string instruct    = json_str(req, "instruct");
        std::string speaker     = json_str(req, "speaker");

        // Helper: write a length-prefixed binary frame to stdout
        auto write_frame = [](const void * data, uint32_t len) {
            if (fwrite(&len, 4, 1, stdout) == 1) {
                if (len > 0) fwrite(data, 1, len, stdout);
            }
            fflush(stdout);
        };

        auto write_error = [&](const std::string & msg) {
            std::string err = "{\"id\":\"" + json_escape(req_id) +
                              "\",\"error\":\"" + json_escape(msg) + "\"}";
            uint32_t st = 1;
            fwrite(&st, 4, 1, stdout);
            write_frame(err.data(), (uint32_t)err.size());
            uint32_t zero = 0;
            fwrite(&zero, 4, 1, stdout);
            fflush(stdout);
        };

        if (req_id.empty() || text.empty()) {
            write_error("missing id or text");
            continue;
        }

        // Load reference audio if changed, and recompute cached voice data
        if (ref_wav != last_ref_wav) {
            int T = 0;
            float * raw = audio_read_mono(ref_wav.c_str(), 24000, &T);
            if (!raw || T <= 0) {
                write_error("cannot read ref_wav: " + ref_wav);
                continue;
            }
            ref_buf.reset(raw);
            ref_n_samples = T;
            last_ref_wav  = ref_wav;

            // Invalidate caches and recompute on the new audio
            cached_spk_emb.clear();
            cached_ref_codes.clear();
            cached_ref_codes_T = 0;

            if (ref_n_samples > 0 && q->pt.has_speaker_encoder) {
                if (!speaker_encoder_extract(&q->pt.speaker_encoder, q->pt.sched,
                                             ref_buf.get(), ref_n_samples,
                                             cached_spk_emb, nullptr)) {
                    fprintf(stderr, "[Loop] WARNING: speaker_encoder_extract failed for %s\n", ref_wav.c_str());
                    cached_spk_emb.clear();
                } else {
                    fprintf(stderr, "[Loop] Cached %zu-dim speaker embedding for %s\n",
                            cached_spk_emb.size(), ref_wav.c_str());
                }
            }

            if (ref_n_samples >= TOKENIZER_HOP_LENGTH) {
                int aligned_T = (ref_n_samples / TOKENIZER_HOP_LENGTH) * TOKENIZER_HOP_LENGTH;
                cached_ref_codes = pipeline_codec_encode(&q->pt.codec, ref_buf.get(), aligned_T, nullptr);
                if (!cached_ref_codes.empty()) {
                    cached_ref_codes_T = (int)cached_ref_codes.size() / q->pt.num_code_groups;
                    fprintf(stderr, "[Loop] Cached %d codec frames for %s\n",
                            cached_ref_codes_T, ref_wav.c_str());
                } else {
                    fprintf(stderr, "[Loop] WARNING: pipeline_codec_encode failed for %s\n", ref_wav.c_str());
                }
            }
        }

        // Load reference text if changed
        if (ref_txt_p != last_ref_text) {
            ref_text_buf.clear();
            if (!ref_txt_p.empty() && !read_text_file(ref_txt_p.c_str(), ref_text_buf)) {
                write_error("cannot read ref_text: " + ref_txt_p);
                continue;
            }
            last_ref_text = ref_txt_p;
        }

        // Build synthesis params
        qt_tts_params params;
        qt_tts_default_params(&params);
        params.text                = text.c_str();
        params.lang                = lang.empty() ? "english" : lang.c_str();
        params.instruct            = instruct.empty() ? nullptr : instruct.c_str();
        params.speaker             = speaker.empty() ? nullptr : speaker.c_str();
        params.ref_audio_24k       = ref_buf.get();
        params.ref_n_samples       = ref_n_samples;
        params.ref_text            = ref_text_buf.empty() ? nullptr : ref_text_buf.c_str();
        params.seed                = json_int64(req, "seed", -1);
        params.max_new_tokens      = (int)json_int64(req, "max_new", 2048);
        params.codec_chunk_sec     = json_float(req, "codec_chunk_sec", 0.5f);
        params.on_chunk            = stream_chunk_cb;

        // Wire in cached voice data so pipeline_tts_synthesize skips recomputation
        if (!cached_spk_emb.empty()) {
            params.ref_spk_emb     = cached_spk_emb.data();
            params.ref_spk_emb_dim = (int)cached_spk_emb.size();
        }
        if (!cached_ref_codes.empty()) {
            params.ref_codes              = cached_ref_codes.data();
            params.ref_codes_T            = cached_ref_codes_T;
            params.ref_codes_num_codebooks = q->pt.num_code_groups;
        }

        StreamCtx sctx = {stdout, true};
        params.on_chunk_user_data = &sctx;

        // Write success header
        uint32_t st = 0;
        fwrite(&st, 4, 1, stdout);
        std::string meta = "{\"id\":\"" + json_escape(req_id) +
                           "\",\"sample_rate\":24000}";
        uint32_t mlen = (uint32_t)meta.size();
        fwrite(&mlen, 4, 1, stdout);
        fwrite(meta.data(), 1, mlen, stdout);

        // Run synthesis (streaming via on_chunk callback)
        qt_audio audio = {};
        qt_status status = qt_synthesize(q, &params, &audio);
        qt_audio_free(&audio);

        // Write zero-length chunk terminator
        uint32_t zero = 0;
        fwrite(&zero, 4, 1, stdout);
        fflush(stdout);

        if (status != QT_STATUS_OK) {
            fprintf(stderr, "[Loop] Synthesis failed: %s\n", qt_last_error());
        } else {
            fprintf(stderr, "[Loop] Synthesized request %s\n", req_id.c_str());
        }
    }

    fprintf(stderr, "[Loop] Exiting\n");
    qt_free(q);
    return 0;
}

static bool parse_args(int argc, char ** argv, Args & a) {
    a                        = {};
    a.lang                   = "english";
    a.format                 = "wav16";
    a.max_new_tokens         = 2048;
    a.seed                   = -1;
    a.do_sample              = true;
    a.temperature            = 0.9f;
    a.top_k                  = 50;
    a.top_p                  = 1.0f;
    a.repetition_penalty     = 1.05f;
    a.subtalker_do_sample    = true;
    a.subtalker_top_k        = 50;
    a.subtalker_top_p        = 1.0f;
    a.subtalker_temperature  = 0.9f;
    a.use_fa                 = true;
    a.clamp_fp16             = false;
    a.codec_chunk_sec        = 24.0f;
    a.codec_left_context_sec = 2.0f;
    a.loop                   = false;
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            return false;
        }
        if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            a.model = argv[++i];
        } else if (std::strcmp(arg, "--codec") == 0 && i + 1 < argc) {
            a.codec = argv[++i];
        } else if (std::strcmp(arg, "--lang") == 0 && i + 1 < argc) {
            a.lang = argv[++i];
        } else if (std::strcmp(arg, "--instruct") == 0 && i + 1 < argc) {
            a.instruct = argv[++i];
        } else if (std::strcmp(arg, "--speaker") == 0 && i + 1 < argc) {
            a.speaker = argv[++i];
        } else if (std::strcmp(arg, "--ref-wav") == 0 && i + 1 < argc) {
            a.ref_wav = argv[++i];
        } else if (std::strcmp(arg, "--ref-text") == 0 && i + 1 < argc) {
            a.ref_text_path = argv[++i];
        } else if (std::strcmp(arg, "--format") == 0 && i + 1 < argc) {
            a.format = argv[++i];
        } else if (std::strcmp(arg, "--dump") == 0 && i + 1 < argc) {
            a.dump_dir = argv[++i];
        } else if (std::strcmp(arg, "--max-new") == 0 && i + 1 < argc) {
            a.max_new_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            a.seed = (int64_t) std::atoll(argv[++i]);
        } else if (std::strcmp(arg, "--greedy") == 0) {
            // Greedy mode: argmax sampling on both stacks. The sampling
            // fast path in sampling.h uses temperature <= 0 to short
            // circuit to argmax, bypassing rep penalty and top-k/p
            // truncation, which exactly mirrors the Python reference
            // greedy behaviour used by tests/debug-tts-cossim.py.
            a.do_sample           = false;
            a.subtalker_do_sample = false;
        } else if (std::strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            a.temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            a.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            a.top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--rep-pen") == 0 && i + 1 < argc) {
            a.repetition_penalty = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-temp") == 0 && i + 1 < argc) {
            a.subtalker_temperature = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-k") == 0 && i + 1 < argc) {
            a.subtalker_top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--sub-top-p") == 0 && i + 1 < argc) {
            a.subtalker_top_p = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--no-fa") == 0) {
            a.use_fa = false;
        } else if (std::strcmp(arg, "--clamp-fp16") == 0) {
            a.clamp_fp16 = true;
        } else if (std::strcmp(arg, "--codec-chunk-dur") == 0 && i + 1 < argc) {
            a.codec_chunk_sec = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--codec-left-dur") == 0 && i + 1 < argc) {
            a.codec_left_context_sec = (float) std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--loop") == 0) {
            a.loop = true;
        } else if (std::strcmp(arg, "-o") == 0 && i + 1 < argc) {
            a.out_wav = argv[++i];
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown or incomplete argument: %s\n", arg);
            return false;
        }
    }
    return a.model && a.codec;
}

static int run(const Args & a) {
    // Init the facade. The seven mode validations
    // (base / custom_voice / voice_design rules) and the BPE tokenizer
    // load live inside qt_init / qt_synthesize; the CLI just hands
    // off the two GGUF paths and reports qt_last_error on failure.
    qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = a.model;
    iparams.codec_path  = a.codec;
    iparams.use_fa      = a.use_fa;
    iparams.clamp_fp16  = a.clamp_fp16;

    qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
        return 1;
    }

    // Load the reference transcript from the file passed via --ref-text.
    // Empty content is rejected since the ICL prompt needs it non empty.
    std::string  ref_text_buf;
    const char * ref_text = NULL;
    if (a.ref_text_path) {
        if (!read_text_file(a.ref_text_path, ref_text_buf)) {
            qt_free(q);
            return 1;
        }
        if (ref_text_buf.empty()) {
            fprintf(stderr, "[CLI] ERROR: --ref-text file '%s' is empty\n", a.ref_text_path);
            qt_free(q);
            return 1;
        }
        ref_text = ref_text_buf.c_str();
    }

    // Decode the reference WAV once, mono at the codec sample rate. The
    // facade consumes the buffer directly so the WAV is read exactly
    // once regardless of how many encoders need the audio (speaker
    // encoder embedding + codec encoder RVQ codes for ICL mode B).
    std::unique_ptr<float, void (*)(void *)> raw_holder(NULL, std::free);
    const float *                            ref_audio_24k = NULL;
    int                                      ref_n_samples = 0;
    if (a.ref_wav) {
        int     T_in = 0;
        float * raw  = audio_read_mono(a.ref_wav, 24000, &T_in);
        if (!raw || T_in <= 0) {
            fprintf(stderr, "[CLI] ERROR: cannot read --ref-wav '%s'\n", a.ref_wav);
            if (raw) {
                std::free(raw);
            }
            qt_free(q);
            return 1;
        }
        raw_holder.reset(raw);
        ref_audio_24k = raw;
        ref_n_samples = T_in;
    }

    // Resolve output WAV format string: wav16 / wav24 / wav32. Default
    // wav16 mirrors the omnivoice.cpp default.
    WavFormat wav_fmt;
    if (!audio_parse_format(a.format, wav_fmt)) {
        fprintf(stderr, "[CLI] ERROR: invalid --format '%s' (expected wav16, wav24, wav32)\n", a.format);
        qt_free(q);
        return 1;
    }

    // Resolve utterance text: read stdin fully. Empty stdin triggers a
    // clean error. Inline text on the command line is intentionally not
    // supported: shell quoting and special characters belong in a file
    // or piped input, never in argv.
    std::string text_buf = read_stdin_text();
    if (text_buf.empty()) {
        fprintf(stderr, "[CLI] ERROR: stdin is empty, nothing to synthesise\n");
        qt_free(q);
        return 1;
    }
    const char * text = text_buf.c_str();

    // Translate CLI args into the facade params. Seed -1 is forwarded
    // verbatim and resolved by qt_synthesize via std::random_device.
    qt_tts_params params;
    qt_tts_default_params(&params);
    params.text                   = text;
    params.lang                   = a.lang;
    params.instruct               = a.instruct;
    params.speaker                = a.speaker;
    params.ref_audio_24k          = ref_audio_24k;
    params.ref_n_samples          = ref_n_samples;
    params.ref_text               = ref_text;
    params.seed                   = a.seed;
    params.max_new_tokens         = a.max_new_tokens;
    params.do_sample              = a.do_sample;
    params.temperature            = a.temperature;
    params.top_k                  = a.top_k;
    params.top_p                  = a.top_p;
    params.repetition_penalty     = a.repetition_penalty;
    params.subtalker_do_sample    = a.subtalker_do_sample;
    params.subtalker_temperature  = a.subtalker_temperature;
    params.subtalker_top_k        = a.subtalker_top_k;
    params.subtalker_top_p        = a.subtalker_top_p;
    params.dump_dir               = a.dump_dir;
    params.codec_chunk_sec        = a.codec_chunk_sec;
    params.codec_left_context_sec = a.codec_left_context_sec;

    // Streaming detection : -o '-' writes a wide RIFF header to stdout
    // up front and pipes encoded samples chunk by chunk through the
    // on_chunk callback as the AR loop produces frames. Any other path
    // (or no -o) takes the buffered route so the file gets accurate
    // sizes in its header.
    const char * out_path         = a.out_wav ? a.out_wav : "out.wav";
    const bool   stream_to_stdout = (out_path[0] == '-' && out_path[1] == '\0');

    if (stream_to_stdout) {
        wav_stream ws = {};
        if (!wav_stream_open_stdout(&ws, 24000, wav_fmt)) {
            qt_free(q);
            return 1;
        }
        params.on_chunk = [](const float * s, int n, void * ud) -> bool {
            return wav_stream_write((wav_stream *) ud, s, n);
        };
        params.on_chunk_user_data = &ws;

        qt_audio  audio  = {};
        qt_status status = qt_synthesize(q, &params, &audio);
        wav_stream_close(&ws);
        if (status != QT_STATUS_OK) {
            fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
            qt_audio_free(&audio);
            qt_free(q);
            return 1;
        }
        qt_audio_free(&audio);
        qt_free(q);
        fprintf(stderr, "[Pipeline] Streamed to <stdout>\n");
        return 0;
    }

    qt_audio  audio  = {};
    qt_status status = qt_synthesize(q, &params, &audio);
    if (status != QT_STATUS_OK) {
        fprintf(stderr, "[CLI] ERROR: %s\n", qt_last_error());
        qt_audio_free(&audio);
        qt_free(q);
        return 1;
    }

    if (audio.n_samples > 0) {
        if (!audio_write_wav(out_path, audio.samples, audio.n_samples, audio.sample_rate, wav_fmt)) {
            fprintf(stderr, "[Pipeline] FATAL: WAV write failed for %s\n", out_path);
            qt_audio_free(&audio);
            qt_free(q);
            return 1;
        }
        fprintf(stderr, "[Pipeline] Wrote %d samples (%.2f s) -> %s\n", audio.n_samples,
                (double) audio.n_samples / (double) audio.sample_rate, out_path);
    }

    qt_audio_free(&audio);
    qt_free(q);
    return 0;
}

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage(argv[0]);
        return 1;
    }
    // The facade absorbs every std::exception thrown deep in the load
    // and synthesis chains, converting them into qt_status + a
    // qt_last_error message. No top-level try / catch needed here :
    // the CLI just reads the status returned by qt_init /
    // qt_synthesize and renders qt_last_error to stderr.
    if (a.loop) {
        return run_loop(a);
    }
    return run(a);
}
