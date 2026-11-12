# p25voc, the engine: IMBE frames to 16 kHz speech with the Vocos-IMBE network (ONNX Runtime) and
# its excitation-phase head and ISTFT in C (KissFFT). From the p25_vocoder project's engine/.
#
# `make` builds it as the plugin ships it, in build/<arch>/: p25voc, and lib/ with libimbe.so
# (op25's IMBE decoder, parameters only) and libonnxruntime.so.1, found through RPATH $ORIGIN/lib.
# Dependencies are fetched into third_party/ first (fetch_deps.sh). For a Raspberry Pi from x86-64:
#   make CROSS=aarch64-linux-gnu-
CROSS  ?=
ARCH    = $(if $(CROSS),$(patsubst %-linux-gnu-,%,$(CROSS)),$(shell uname -m))
TP      = third_party
ORT     = $(TP)/onnxruntime$(if $(CROSS),-$(ARCH))
KISS    = $(TP)/kissfft
IMBE    = $(TP)/imbe_vocoder
BUILD   = build/$(ARCH)

CC      = $(CROSS)gcc
CXX     = $(CROSS)g++
CFLAGS  = -O3 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I$(ORT)/include -I$(KISS) -Dkiss_fft_scalar=float
CXXFLAGS = -O2 -fPIC -I$(IMBE) -std=c++11 -w
IMBE_SRCS = $(addprefix $(IMBE)/, imbe_vocoder.cc aux_sub.cc basicop2.cc ch_decode.cc ch_encode.cc dc_rmv.cc \
	decode.cc dsp_sub.cc encode.cc math_sub.cc pe_lpf.cc pitch_est.cc pitch_ref.cc qnt_sub.cc rand_gen.cc \
	sa_decode.cc sa_encode.cc sa_enh.cc tbls.cc uv_synt.cc v_synt.cc v_uv_det.cc)

$(BUILD)/p25voc: p25voc.c $(BUILD)/lib/libimbe.so $(BUILD)/lib/libonnxruntime.so.1
	$(CC) $(CFLAGS) -o $@ p25voc.c $(KISS)/kiss_fft.c $(KISS)/kiss_fftr.c -L$(BUILD)/lib \
		-Wl,-rpath,'$$ORIGIN/lib' -l:libonnxruntime.so.1 -limbe -lm

$(BUILD)/lib/libimbe.so: imbe_capi.cpp | deps
	mkdir -p $(BUILD)/lib
	$(CXX) $(CXXFLAGS) -shared -o $@ $(IMBE_SRCS) imbe_capi.cpp -lm

$(BUILD)/lib/libonnxruntime.so.1: | deps
	mkdir -p $(BUILD)/lib
	cp -L $(ORT)/lib/libonnxruntime.so.1 $@

deps:
	ARCH=$(ARCH) bash fetch_deps.sh

clean:
	rm -rf build

.PHONY: deps clean
