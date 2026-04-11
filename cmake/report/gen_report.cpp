/// @file report/gen_report.cpp
/// @brief Markdown report assembler.
///
/// Usage:
///   gen_report <template.md> <out.md> <test.json> <bench.json> <build_info.json> <plots_dir>
///
/// Reads template.md, substitutes every {{KEY}} with generated markdown, writes out.md.
/// Missing data (backend absent, JSON not found) leaves a *[not available]* placeholder.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// --- Minimal JSON helpers ----------------------------------------------------

/// Read entire file into string. Returns "" on error.
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    return {std::istreambuf_iterator<char>(f), {}};
}

/// Find the content of array "key" in json (between the matching [ ... ]).
static std::string jarray(const std::string& json, const std::string& key) {
    std::string kq = "\"" + key + "\"";
    size_t pos = json.find(kq);
    if (pos == json.npos) return "";
    pos = json.find('[', pos + kq.size());
    if (pos == json.npos) return "";
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < json.size(); ++i) {
        if (json[i] == '[') ++depth;
        else if (json[i] == ']' && --depth == 0)
            return json.substr(start + 1, i - start - 1);
    }
    return "";
}

/// Split top-level { ... } objects from an array content string.
static std::vector<std::string> split_objects(const std::string& s) {
    std::vector<std::string> result;
    int depth = 0;
    size_t start = 0;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_str = !in_str;
        if (in_str) continue;
        if (c == '{') { if (depth++ == 0) start = i; }
        else if (c == '}' && --depth == 0)
            result.push_back(s.substr(start, i - start + 1));
    }
    return result;
}

/// Extract string value of key from a flat JSON object string.
static std::string jstr(const std::string& obj, const std::string& key) {
    std::string kq = "\"" + key + "\"";
    size_t pos = obj.find(kq);
    if (pos == obj.npos) return "";
    pos = obj.find('"', pos + kq.size());
    if (pos == obj.npos) return "";
    size_t start = pos + 1, end = start;
    while (end < obj.size() && !(obj[end] == '"' && obj[end - 1] != '\\')) ++end;
    return obj.substr(start, end - start);
}

/// Extract numeric value of key from a flat JSON object string (first occurrence).
static double jnum(const std::string& obj, const std::string& key) {
    std::string kq = "\"" + key + "\"";
    size_t pos = obj.find(kq);
    if (pos == obj.npos) return 0.0;
    // find next number (skip : whitespace)
    size_t np = pos + kq.size();
    while (np < obj.size() && obj[np] != '-' && !isdigit(obj[np])) {
        if (obj[np] == '"' || obj[np] == '{' || obj[np] == '[') return 0.0;
        ++np;
    }
    if (np >= obj.size()) return 0.0;
    size_t end = np + 1;
    while (end < obj.size() && (isdigit(obj[end]) || obj[end] == '.'
            || obj[end] == 'e' || obj[end] == 'E'
            || obj[end] == '+' || obj[end] == '-')) ++end;
    try { return std::stod(obj.substr(np, end - np)); } catch (...) { return 0.0; }
}

/// Extract integer value of key.
static int jint(const std::string& obj, const std::string& key) {
    return static_cast<int>(jnum(obj, key));
}

/// Extract bool value ("true"/"false") of key.
static bool jbool(const std::string& obj, const std::string& key) {
    std::string kq = "\"" + key + "\"";
    size_t pos = obj.find(kq);
    if (pos == obj.npos) return false;
    pos = obj.find_first_not_of(" \t\r\n:", pos + kq.size());
    if (pos == obj.npos) return false;
    return obj.substr(pos, 4) == "true";
}

// --- Test result types --------------------------------------------------------

struct TestSuite {
    std::string name;
    int total = 0, failures = 0;
    double time_ms = 0.0;
};

static std::vector<TestSuite> parse_gtest_json(const std::string& path) {
    std::string raw = read_file(path);
    if (raw.empty()) return {};
    std::vector<TestSuite> result;
    std::string arr = jarray(raw, "testsuites");
    for (auto& obj : split_objects(arr)) {
        TestSuite s;
        s.name     = jstr(obj, "name");
        s.total    = jint(obj, "tests");
        s.failures = jint(obj, "failures");
        // time: "0.123s"
        std::string t = jstr(obj, "time");
        if (!t.empty() && t.back() == 's')
            try { s.time_ms = std::stod(t.substr(0, t.size() - 1)) * 1000.0; }
            catch (...) {}
        if (!s.name.empty()) result.push_back(s);
    }
    return result;
}

// --- Benchmark result types ---------------------------------------------------

struct BenchRun {
    std::string name, run_type, time_unit;
    double real_time = 0.0;   // in time_unit
    double bps       = 0.0;   // bytes_per_second
};

static double to_us(double t, const std::string& unit) {
    if (unit == "ns") return t / 1000.0;
    if (unit == "ms") return t * 1000.0;
    if (unit == "s")  return t * 1e6;
    return t; // "us"
}

static std::vector<BenchRun> parse_bench_json(const std::string& path) {
    std::string raw = read_file(path);
    if (raw.empty()) return {};
    std::vector<BenchRun> result;
    std::string arr = jarray(raw, "benchmarks");
    for (auto& obj : split_objects(arr)) {
        BenchRun r;
        r.name      = jstr(obj, "name");
        r.run_type  = jstr(obj, "run_type");
        r.time_unit = jstr(obj, "time_unit");
        r.real_time = jnum(obj, "real_time");
        r.bps       = jnum(obj, "bytes_per_second");
        if (!r.name.empty()) result.push_back(r);
    }
    return result;
}

// --- Build info types ---------------------------------------------------------

struct BuildInfo {
    std::string date, compiler, build_type;
    bool has_blas = false, has_fftw = false, has_omp = false;
    bool has_cuda = false, has_mpi  = false, has_lapack = false;
    std::string system_name, cpu, ram_mb, gpu;
};

static BuildInfo parse_build_info(const std::string& path) {
    std::string raw = read_file(path);
    BuildInfo b;
    if (raw.empty()) return b;
    b.date        = jstr(raw, "date");
    b.compiler    = jstr(raw, "compiler");
    b.build_type  = jstr(raw, "build_type");
    b.has_blas    = jbool(raw, "has_blas");
    b.has_fftw    = jbool(raw, "has_fftw");
    b.has_omp     = jbool(raw, "has_omp");
    b.has_cuda    = jbool(raw, "has_cuda");
    b.has_mpi     = jbool(raw, "has_mpi");
    b.has_lapack  = jbool(raw, "has_lapack");
    b.system_name = jstr(raw, "system_name");
    b.cpu         = jstr(raw, "cpu");
    b.ram_mb      = jstr(raw, "ram_mb");
    b.gpu         = jstr(raw, "gpu");
    return b;
}

// --- Markdown generators ------------------------------------------------------

static std::string system_info_table(const BuildInfo& b) {
    auto ram_str = [&]() -> std::string {
        if (b.ram_mb.empty()) return "n/a";
        try {
            long mb = std::stol(b.ram_mb);
            char buf[32];
            snprintf(buf, sizeof(buf), "%ld MB (%.1f GB)", mb, mb / 1024.0);
            return buf;
        } catch (...) { return b.ram_mb + " MB"; }
    };
    std::string s =
        "| Property | Value |\n"
        "|----------|-------|\n";
    s += "| OS       | " + (b.system_name.empty() ? "n/a" : b.system_name) + " |\n";
    s += "| CPU      | " + (b.cpu.empty()         ? "n/a" : b.cpu)         + " |\n";
    s += "| RAM      | " + ram_str()                                         + " |\n";
    s += "| GPU      | " + (b.gpu.empty()         ? "n/a" : b.gpu)         + " |\n";
    return s;
}

static std::string backends_table(const BuildInfo& b) {
    std::string s =
        "| Backend | Status | Notes |\n"
        "|---------|--------|-------|\n";
    auto row = [&](const char* name, bool found, const char* note) {
        s += std::string("| ") + name + " | "
          + (found ? "**found**" : "not found") + " | " + note + " |\n";
    };
    row("BLAS / cblas",  b.has_blas,   "Backend::blas   -- cblas_dgemm, cblas_ddot, cblas_dgemv");
    row("LAPACKE",       b.has_lapack, "Backend::lapack -- dgetrf, dgeqrf, dgesdd, dsyevd, dgtsv");
    row("OpenMP",        b.has_omp,    "Backend::omp    -- parallel blocked loops");
    row("FFTW3",         b.has_fftw,   "FFTBackend::fftw -- AVX2/NEON optimised DFT");
    row("CUDA",          b.has_cuda,   "Backend::gpu    -- custom kernels / cuBLAS");
    row("MPI",           b.has_mpi,    "distributed ops (experimental)");
    return s;
}

/// Build a markdown table for a subset of test suites.
/// suite_filter: if non-empty, only include suites whose name is in the set.
static std::string tests_table(const std::vector<TestSuite>& suites,
                                const std::vector<std::string>& filter = {}) {
    std::vector<const TestSuite*> rows;
    for (auto& s : suites) {
        if (filter.empty()) { rows.push_back(&s); continue; }
        for (auto& f : filter)
            if (s.name == f) { rows.push_back(&s); break; }
    }
    if (rows.empty()) return "*No test data -- run `make report` to generate.*\n";

    std::string t =
        "| Suite | Tests | Passed | Failed | Time |\n"
        "|-------|------:|-------:|-------:|-----:|\n";
    for (auto* s : rows) {
        int passed = s->total - s->failures;
        char buf[256];
        snprintf(buf, sizeof(buf), "| %s | %d | %d | %d | %.1f ms |\n",
                 s->name.c_str(), s->total, passed, s->failures, s->time_ms);
        t += buf;
    }
    return t;
}

/// Extract size from benchmark name (number after last '/').
static long bench_size(const std::string& name) {
    auto pos = name.rfind('/');
    if (pos == name.npos) return -1;
    try { return std::stol(name.substr(pos + 1)); } catch (...) { return -1; }
}

/// Map a raw benchmark name to a short human-readable variant label.
static std::string bench_label(const std::string& name) {
    auto has = [&](const char* s){ return name.find(s) != name.npos; };
    // FFT-specific backends first (more specific substrings before generic ones)
    if (has("FFTBackend::stdsimd"))  return "std::simd";
    if (has("FFTBackend::simd"))     return "simd (AVX2/NEON)";
    if (has("FFTBackend::fftw"))     return "fftw (FFTW3)";
    if (has("FFTBackend::seq"))      return "seq (Cooley-Tukey)";
    // Linalg backends
    if (has("Backend::blas"))        return "blas";
    if (has("Backend::omp"))         return "omp";
    if (has("Backend::simd"))        return "simd";
    if (has("Backend::blocked"))     return "blocked";
    if (has("Backend::seq"))         return "seq";
    if (has("Matmul_RegBlocked"))    return "reg-blocked";
    if (has("Matmul_Blocked"))       return "blocked (auto-vec)";
    if (has("Matmul_Naive"))         return "naive";
    if (has("CG_GPU") || has("GPU")) return "gpu";
    // fallback: strip BM_ prefix and /N suffix
    std::string v = name;
    if (v.size() > 3 && v.substr(0,3) == "BM_") v = v.substr(3);
    auto sl = v.rfind('/');
    if (sl != v.npos) v = v.substr(0, sl);
    return v;
}

/// Generate a markdown benchmark table.
/// @param prefix   Substring that must appear in benchmark name.
/// @param use_bps  true = GB/s from bytes_per_second;  false = us from real_time.
static std::string bench_table(const std::vector<BenchRun>& all,
                                 const std::string& prefix,
                                 bool use_bps,
                                 const std::string& footer = "") {
    // Collect data
    std::map<std::string, std::map<long, double>> data; // label -> size -> value
    std::set<long> all_sizes;
    std::vector<std::string> order; // preserve first-seen label order

    for (auto& r : all) {
        if (r.run_type != "iteration") continue;
        if (r.name.find("_BigO") != r.name.npos) continue;
        if (r.name.find("_RMS")  != r.name.npos) continue;
        if (r.name.find(prefix)  == r.name.npos) continue;
        long sz = bench_size(r.name);
        if (sz <= 0) continue;
        std::string lbl = bench_label(r.name);
        double val = use_bps ? r.bps / 1e9
                              : to_us(r.real_time, r.time_unit);
        if (data.find(lbl) == data.end()) order.push_back(lbl);
        data[lbl][sz] = val;
        all_sizes.insert(sz);
    }

    if (data.empty()) return "*No benchmark data -- run `make report` to generate.*\n";

    std::vector<long> sizes(all_sizes.begin(), all_sizes.end());

    // Header row
    std::string unit_hdr = use_bps ? " GB/s" : " us";
    std::string t = "| Variant |";
    std::string sep = "|---------|";
    for (long n : sizes) {
        std::string col = " n=" + std::to_string(n) + unit_hdr + " |";
        t   += col;
        sep += std::string(col.size() - 1, '-') + "|";
    }
    t += "\n" + sep + "\n";

    // Data rows
    for (auto& lbl : order) {
        t += "| " + lbl + " |";
        for (long n : sizes) {
            auto it = data[lbl].find(n);
            if (it != data[lbl].end()) {
                char buf[32];
                if (use_bps) snprintf(buf, sizeof(buf), " %.2f |", it->second);
                else         snprintf(buf, sizeof(buf), " %.1f |", it->second);
                t += buf;
            } else {
                t += " -- |";
            }
        }
        t += "\n";
    }

    if (!footer.empty()) t += "\n" + footer + "\n";
    return t;
}

/// Return markdown image tag if the file exists, else a placeholder.
static std::string figure(const std::string& plots_dir,
                           const std::string& filename,
                           const std::string& alt) {
    std::string p = plots_dir + "/" + filename;
    if (fs::exists(p))
        return "!["  + alt + "](plots/" + filename + ")\n";
    return "*Plot not available (backend absent or benchmarks not run).*\n";
}


// --- Substitution map builder -------------------------------------------------

static std::map<std::string, std::string> build_subs(
        const BuildInfo& info,
        const std::vector<TestSuite>& tests,
        const std::vector<BenchRun>& bench,
        const std::string& plots_dir) {

    std::map<std::string, std::string> s;

    // --- Build metadata ---
    s["BUILD_DATE"]     = info.date;
    s["COMPILER"]       = info.compiler;
    s["BUILD_TYPE"]     = info.build_type;
    s["SYSTEM_INFO"]    = system_info_table(info);
    s["BACKENDS_TABLE"] = backends_table(info);

    // --- Test tables ---
    s["TESTS_SUMMARY"]      = tests_table(tests);
    s["TESTS_CORE"]         = tests_table(tests, {"Vector", "Matrix"});
    s["TESTS_FACTORIZATION"]= tests_table(tests, {"LU", "QR", "Thomas", "TriDiag"});
    s["TESTS_SOLVERS"]      = tests_table(tests, {"CG", "GaussSeidel", "Jacobi", "Krylov", "Solver"});
    s["TESTS_BANDED"]       = tests_table(tests, {"Banded"});
    s["TESTS_ANALYSIS"]     = tests_table(tests, {"Roots", "Quadrature", "Analysis"});
    s["TESTS_FFT"]          = tests_table(tests, {"FFT", "FFTPlan"});
    s["TESTS_EIGEN"]        = tests_table(tests, {"EigSym", "PowerIteration", "Lanczos"});
    s["TESTS_SVD"]          = tests_table(tests, {"SVD"});

    // --- Benchmark tables ---
    // Core
    s["BENCH_MATMUL_TABLE"] = bench_table(bench, "BM_Matmul", false,
        "*Time in us. Lower is better.*");
    s["BENCH_MATVEC_TABLE"] = bench_table(bench, "BM_Matvec", true,
        "*Throughput in GB/s. Higher is better.*");
    s["BENCH_DOT_TABLE"]    = bench_table(bench, "BM_Dot",    true,
        "*Throughput in GB/s. Higher is better.*");
    s["BENCH_AXPY_TABLE"]   = bench_table(bench, "BM_Axpy",   true,
        "*Throughput in GB/s. Higher is better.*");
    // Linalg
    s["BENCH_CG_TABLE"]     = bench_table(bench, "BM_CG",     false,
        "*Time in us. Lower is better.*");
    s["BENCH_THOMAS_TABLE"] = bench_table(bench, "BM_Thomas/", false,
        "*Time in us. Linear O(n) scaling expected.*");
    s["BENCH_BANDED_TABLE"] = bench_table(bench, "BM_Band",   false,
        "*Time in us. Lower is better.*");
    // Factorization (3-way: seq vs omp vs LAPACK)
    s["BENCH_LU_TABLE"]        = bench_table(bench, "BM_LU",            false,
        "*GFLOP/s: 2/3 n^3 / time. Higher is better.*");
    s["BENCH_QR_TABLE"]        = bench_table(bench, "BM_QR",            false,
        "*GFLOP/s: 4/3 n^3 / time. Higher is better.*");
    // Eigensolvers (3-way: Jacobi seq vs omp vs LAPACK; plus Lanczos)
    s["BENCH_EIGSYM_TABLE"]    = bench_table(bench, "BM_EigSym",        false,
        "*Time in us. Lower is better.*");
    s["BENCH_LANCZOS_TABLE"]   = bench_table(bench, "BM_Lanczos",       false,
        "*Time in us (k=10 eigenvalues). Lower is better.*");
    // SVD (3-way: Jacobi vs randomized vs LAPACK)
    s["BENCH_SVD_TABLE"]       = bench_table(bench, "BM_SVD/",          false,
        "*Time in us. Lower is better.*");
    s["BENCH_SVD_RAND_TABLE"]  = bench_table(bench, "BM_SVD_Randomized",false,
        "*Time in us (k=n/8 singular values). Lower is better.*");
    // Spectral
    s["BENCH_FFT_TABLE"]      = bench_table(bench, "BM_FFT",     true,
        "*Throughput in GB/s. Higher is better.*");
    s["BENCH_FFT_PLAN_TABLE"] = bench_table(bench, "BM_FFTPlan", true,
        "*Throughput in GB/s. Plan creation excluded. Higher is better.*");
    s["BENCH_RFFT_TABLE"]     = bench_table(bench, "BM_RFFT",    true,
        "*Throughput in GB/s. Higher is better.*");

    // --- Figures (PNG) ---
    auto fig = [&](const std::string& key, const std::string& file, const std::string& alt){
        s[key] = figure(plots_dir, file, alt);
    };
    fig("BENCH_MATMUL_FIGURE",   "matmul.png",   "Matrix multiply: GFLOP/s vs n");
    fig("BENCH_MATVEC_FIGURE",   "matvec.png",   "Matrix-vector multiply: GB/s vs n");
    fig("BENCH_DOT_FIGURE",      "dot.png",      "Dot product: GB/s vs n");
    fig("BENCH_AXPY_FIGURE",     "axpy.png",     "Axpy: GB/s vs n");
    fig("BENCH_CG_FIGURE",       "cg.png",       "Conjugate gradient: time vs n");
    fig("BENCH_THOMAS_FIGURE",   "thomas.png",   "Thomas algorithm: time vs n");
    fig("BENCH_BANDED_FIGURE",   "banded.png",   "Banded solver: time vs n");
    fig("BENCH_LU_FIGURE",       "lu.png",       "LU factorization: GFLOP/s vs n");
    fig("BENCH_QR_FIGURE",       "qr.png",       "QR factorization: GFLOP/s vs n");
    fig("BENCH_EIGSYM_FIGURE",   "eigen.png",    "Symmetric eigensolver: time vs n");
    fig("BENCH_SVD_FIGURE",      "svd.png",      "SVD: time vs n");
    fig("BENCH_FFT_FIGURE",      "fft.png",      "FFT throughput: GB/s vs n");
    fig("BENCH_FFT_PLAN_FIGURE", "fft_plan.png", "FFT plan throughput: GB/s vs n");
    fig("BENCH_RFFT_FIGURE",     "rfft.png",     "rFFT throughput: GB/s vs n");

    return s;
}

// --- HTML generator -----------------------------------------------------------

/// JSON-encode a string for safe embedding inside a <script> tag.
/// Escapes \, ", control chars, and / (to prevent </script> injection).
static std::string js_encode(const std::string& s) {
    std::string r;
    r.reserve(s.size() + 64);
    r += '"';
    for (unsigned char c : s) {
        if      (c == '"')  r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '/')  r += "\\/";   // prevents </script> in content
        else if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else if (c == '\t') r += "\\t";
        else if (c < 0x20) { char buf[8]; snprintf(buf,sizeof(buf),"\\u%04x",c); r+=buf; }
        else r += static_cast<char>(c);
    }
    r += '"';
    return r;
}

/// Wrap a rendered markdown report in a minimal self-contained HTML page.
/// The markdown is embedded as a JS string — no network fetch required.
static std::string gen_html_report(const std::string& markdown,
                                    const std::string& build_date) {
    std::string md_js = js_encode(markdown);
    return
R"(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>numerics — Status Report ()" + build_date + R"()</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    body { max-width: 900px; margin: 0 auto; padding: 2rem; font-family: sans-serif; line-height: 1.6; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9em; }
    th, td { border: 1px solid #ccc; padding: .4rem .7rem; text-align: left; }
    th { background: #f4f4f4; }
    img { max-width: 100%; display: block; margin: .75rem 0; }
    pre { background: #f6f6f6; padding: 1rem; overflow-x: auto; }
    code { background: #f6f6f6; padding: .1em .3em; }
    pre code { background: none; padding: 0; }
    a { color: #0969da; }
  </style>
</head>
<body>
<div id="content"></div>
<script>
(function(){
  var md = )" + md_js + R"(;
  var html = marked.parse(md);
  // Inject id attributes into headings so TOC anchor links work.
  // Algorithm mirrors GitHub: lowercase, strip non-(alnum|space|hyphen), spaces→hyphens.
  html = html.replace(/<(h[1-6])>(.*?)<\/\1>/g, function(_, tag, inner) {
    var plain = inner.replace(/<[^>]+>/g, '');
    var id = plain.toLowerCase().replace(/[^a-z0-9 -]/g, '').replace(/ /g, '-');
    return '<' + tag + ' id="' + id + '">' + inner + '</' + tag + '>';
  });
  document.getElementById('content').innerHTML = html;
})();
</script>
</body>
</html>
)";
}

// --- Template engine ----------------------------------------------------------

static std::string apply_template(const std::string& tmpl,
                                   const std::map<std::string, std::string>& subs) {
    std::string out;
    out.reserve(tmpl.size() * 2);
    size_t i = 0;
    while (i < tmpl.size()) {
        if (tmpl[i] == '{' && i + 1 < tmpl.size() && tmpl[i + 1] == '{') {
            size_t end = tmpl.find("}}", i + 2);
            if (end != tmpl.npos) {
                std::string key = tmpl.substr(i + 2, end - i - 2);
                auto it = subs.find(key);
                out += (it != subs.end()) ? it->second
                                          : "*[" + key + " -- data not available]*";
                i = end + 2;
                continue;
            }
        }
        out += tmpl[i++];
    }
    return out;
}

// --- main ---------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: gen_report <template.md> <out.md> "
                     "<test.json> <bench.json> <build_info.json> <plots_dir>\n";
        return 1;
    }
    std::string tmpl_path  = argv[1];
    std::string out_path   = argv[2];
    std::string test_path  = argv[3];
    std::string bench_path = argv[4];
    std::string info_path  = argv[5];
    std::string plots_dir  = argv[6];

    // Load template
    std::string tmpl = read_file(tmpl_path);
    if (tmpl.empty()) {
        std::cerr << "gen_report: cannot read template " << tmpl_path << "\n";
        return 1;
    }

    // Parse data
    auto info  = parse_build_info(bench_path.empty() ? "" : info_path);
    auto tests = parse_gtest_json(test_path);
    auto bench = parse_bench_json(bench_path);

    // Build substitutions
    auto subs = build_subs(info, tests, bench, plots_dir);

    // Apply
    std::string report = apply_template(tmpl, subs);

    // Write markdown
    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "gen_report: cannot write " << out_path << "\n";
        return 1;
    }
    out << report;
    std::cout << "report written to " << out_path << "\n";

    // Write self-contained HTML (replace .md extension with .html)
    std::string html_path = out_path;
    auto dot = html_path.rfind('.');
    if (dot != html_path.npos) html_path = html_path.substr(0, dot);
    html_path += ".html";

    std::ofstream html_out(html_path);
    if (html_out) {
        html_out << gen_html_report(report, info.date);
        std::cout << "HTML report written to " << html_path << "\n";
    } else {
        std::cerr << "gen_report: warning: cannot write " << html_path << "\n";
    }

    return 0;
}
