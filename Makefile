# espRL: RL policies trained on the Mac, run as C on an ESP32-S3 (Arduino Nano ESP32) over USB.
IDF  := tools/idf.sh
PY   := .venv/bin/python
# Arduino Nano ESP32. USB is not passed into the IDF container, so flash from the host.
PORT ?= $(firstword $(wildcard /dev/cu.usbmodem*))

.PHONY: setup test train fw flash hil viz

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

test:
	$(PY) -m pytest -q

train:
	$(PY) train.py

fw:
	$(IDF) firmware -B build-esp32s3 -D SDKCONFIG=build-esp32s3/sdkconfig -D IDF_TARGET=esp32s3 build

flash:
	# One retry: the first USB reset into download mode fails about every other time while the app runs.
	cd firmware/build-esp32s3 && for i in 1 2; do uvx esptool --chip esp32s3 -p $(PORT) -b 460800 --before usb-reset \
		write-flash @flash_args && exit 0; sleep 1; done; exit 1

hil:
	$(PY) hil.py $(PORT)

# Live window; VIZ_ARGS="--save out/hil.gif" records one episode instead.
viz:
	$(PY) viz.py $(PORT) $(VIZ_ARGS)
