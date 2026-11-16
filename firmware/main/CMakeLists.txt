idf_component_register(SRCS "main.c" "../../mlp.c"
                       INCLUDE_DIRS "." "../.."
                       PRIV_REQUIRES esp_driver_usb_serial_jtag)
# Inference time is a reported result: our sources are built -O2. The whole image at -O2
# (CONFIG_COMPILER_OPTIMIZATION_PERF) answered one request per port open, then went silent.
set_source_files_properties(../../mlp.c main.c PROPERTIES COMPILE_OPTIONS "-O2")
