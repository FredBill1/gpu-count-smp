/*
 * This file is modified from https://github.com/aminnj/cpptqdm/blob/04c733fd38cdc1763d7bc19f8ff3a8fb6e95e2e9/tqdm.h
 * Original author: Nick Amin
 *
 * Modifications:
 * - Modified bar themes to ensure MSVC compatibility
 *
 * Original MIT License:
 * MIT License
 *
 * Copyright (c) 2018 Nick Amin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef TQDM_H
#define TQDM_H
#if _MSC_VER
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif
#include <math.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

class tqdm {
   private:
    // time, iteration counters and deques for rate calculations
    std::chrono::time_point<std::chrono::system_clock> t_first = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> t_old = std::chrono::system_clock::now();
    int n_old = 0;
    std::vector<double> deq_t;
    std::vector<int> deq_n;
    int nupdates = 0;
    int total_ = 0;
    int period = 1;
    unsigned int smoothing = 50;
    bool use_ema = true;
    float alpha_ema = 0.1;

    // std::vector<const char*> bars = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
    std::vector<const char*> bars = {" ",
                                     "\xE2\x96\x8F",
                                     "\xE2\x96\x8E",
                                     "\xE2\x96\x8D",
                                     "\xE2\x96\x8C",
                                     "\xE2\x96\x8B",
                                     "\xE2\x96\x8A",
                                     "\xE2\x96\x89",
                                     "\xE2\x96\x88"};

#if _MSC_VER
    bool in_screen = false, in_tmux = false;
#else
    bool in_screen = (system("test $STY") == 0);
    bool in_tmux = (system("test $TMUX") == 0);
#endif
    bool is_tty;
    bool use_colors = true;
    bool color_transition = true;
    int width = 40;

    std::string right_pad = "\xE2\x96\x8F";
    std::string label = "";

    void hsv_to_rgb(float h, float s, float v, int& r, int& g, int& b) {
        if (s < 1e-6) {
            v *= 255.;
            r = v;
            g = v;
            b = v;
        }
        int i = (int)(h * 6.0);
        float f = (h * 6.) - i;
        int p = (int)(255.0 * (v * (1. - s)));
        int q = (int)(255.0 * (v * (1. - s * f)));
        int t = (int)(255.0 * (v * (1. - s * (1. - f))));
        v *= 255;
        i %= 6;
        int vi = (int)v;
        if (i == 0) {
            r = vi;
            g = t;
            b = p;
        } else if (i == 1) {
            r = q;
            g = vi;
            b = p;
        } else if (i == 2) {
            r = p;
            g = vi;
            b = t;
        } else if (i == 3) {
            r = p;
            g = q;
            b = vi;
        } else if (i == 4) {
            r = t;
            g = p;
            b = vi;
        } else if (i == 5) {
            r = vi;
            g = p;
            b = q;
        }
    }
    FILE* file;

   public:
    tqdm(FILE* file = stderr) : file(file) {
        is_tty = isatty(fileno(file));
        if (in_screen) {
            set_theme_basic();
            color_transition = false;
        } else if (in_tmux) {
            color_transition = false;
        }
    }

    void reset() {
        t_first = std::chrono::system_clock::now();
        t_old = std::chrono::system_clock::now();
        n_old = 0;
        deq_t.clear();
        deq_n.clear();
        period = 1;
        nupdates = 0;
        total_ = 0;
        label = "";
    }

    // void set_theme_line() { bars = {"─", "─", "─", "╾", "╾", "╾", "╾", "━", "═"}; }
    // void set_theme_circle() { bars = {" ", "◓", "◑", "◒", "◐", "◓", "◑", "◒", "#"}; }
    // void set_theme_braille() { bars = {" ", "⡀", "⡄", "⡆", "⡇", "⡏", "⡟", "⡿", "⣿"}; }
    // void set_theme_braille_spin() { bars = {" ", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠇", "⠿"}; }
    // void set_theme_vertical() { bars = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "█"}; }
    void set_theme_basic() {
        bars = {" ", " ", " ", " ", " ", " ", " ", " ", "#"};
        right_pad = "|";
    }
    void set_label(std::string label_) { label = label_; }
    void disable_colors() {
        color_transition = false;
        use_colors = false;
    }

    void finish() {
        if (!is_tty) return;
        progress(total_, total_);
        fprintf(file, "\n");
        fflush(file);
    }
    void progress(int curr, int tot) {
        if (is_tty && (curr % period == 0)) {
            total_ = tot;
            nupdates++;
            auto now = std::chrono::system_clock::now();
            double dt = ((std::chrono::duration<double>)(now - t_old)).count();
            double dt_tot = ((std::chrono::duration<double>)(now - t_first)).count();
            int dn = curr - n_old;
            n_old = curr;
            t_old = now;
            if (deq_n.size() >= smoothing) deq_n.erase(deq_n.begin());
            if (deq_t.size() >= smoothing) deq_t.erase(deq_t.begin());
            deq_t.push_back(dt);
            deq_n.push_back(dn);

            double avgrate = 0.;
            if (use_ema) {
                avgrate = deq_n[0] / deq_t[0];
                for (unsigned int i = 1; i < deq_t.size(); i++) {
                    double r = 1.0 * deq_n[i] / deq_t[i];
                    avgrate = alpha_ema * r + (1.0 - alpha_ema) * avgrate;
                }
            } else {
                double dtsum = std::accumulate(deq_t.begin(), deq_t.end(), 0.);
                int dnsum = std::accumulate(deq_n.begin(), deq_n.end(), 0.);
                avgrate = dnsum / dtsum;
            }

            // learn an appropriate period length to avoid spamming stdout
            // and slowing down the loop, shoot for ~25Hz and smooth over 3 seconds
            if (nupdates > 10) {
                period = (int)(std::min(std::max((1.0 / 25) * curr / dt_tot, 1.0), 5e5));
                smoothing = 25 * 3;
            }
            double peta = (tot - curr) / avgrate;
            double pct = (double)curr / (tot * 0.01);
            if ((tot - curr) <= period) {
                pct = 100.0;
                avgrate = tot / dt_tot;
                curr = tot;
                peta = 0;
            }

            double fills = ((double)curr / tot * width);
            int ifills = (int)fills;

            fprintf(file, "\015 ");
            if (use_colors) {
                if (color_transition) {
                    // red (hue=0) to green (hue=1/3)
                    int r = 255, g = 255, b = 255;
                    hsv_to_rgb(0.0 + 0.01 * pct / 3, 0.65, 1.0, r, g, b);
                    fprintf(file, "\033[38;2;%d;%d;%dm ", r, g, b);
                } else {
                    fprintf(file, "\033[32m ");
                }
            }
            for (int i = 0; i < ifills; i++) fprintf(file, "%s", bars[8]);
            if (!in_screen && (curr != tot)) fprintf(file, "%s", bars[(int)(8.0 * (fills - ifills))]);
            for (int i = 0; i < width - ifills - 1; i++) fprintf(file, "%s", bars[0]);
            fprintf(file, "%s ", right_pad.c_str());
            if (use_colors) fprintf(file, "\033[1m\033[31m");
            fprintf(file, "%4.1f%% ", pct);
            if (use_colors) fprintf(file, "\033[34m");

            std::string unit = "Hz";
            double div = 1.;
            if (avgrate > 1e6) {
                unit = "MHz";
                div = 1.0e6;
            } else if (avgrate > 1e3) {
                unit = "kHz";
                div = 1.0e3;
            }
            fprintf(file, "[%4d/%4d | %3.1f %s | %.0fs<%.0fs] ", curr, tot, avgrate / div, unit.c_str(), dt_tot, peta);
            fprintf(file, "%s ", label.c_str());
            if (use_colors) fprintf(file, "\033[0m\033[32m\033[0m\015 ");

            if ((tot - curr) > period) fflush(file);
        }
    }
};
#endif
