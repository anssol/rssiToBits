CONTIKI = ../..
CONTIKI_PROJECT = cc1310-rssi-scanner

TARGET=srf06-cc26xx
BOARD=launchpad/cc1310

ifdef CHANNEL
CFLAGS+="-DCHANNEL=$(CHANNEL)"
else ifdef FREQ
CFLAGS+="-DFREQ=$(FREQ)"
endif

ifdef INTERVAL
CFLAGS+="-DINTERVAL=$(INTERVAL)"
endif

UNIFLASH = /opt/ti/uniflash/dslite.sh

all: $(CONTIKI_PROJECT).bin

flash: $(CONTIKI_PROJECT).elf
	$(UNIFLASH) $(CONTIKI_PROJECT).elf -c CC1310F128.ccxml -O PinReset

include $(CONTIKI)$/Makefile.include
