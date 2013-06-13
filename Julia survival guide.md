Here are some things you need to be aware of:
- The Julia binaries may be very old. It is much better to compile Julia from source (available at github).
- The manual usually lags behind the code.
- Julia does not have a mechanism to clear variables, constants, types , functions. If you are using Julia from the REPL (the console), it is easiest to just restart the REPL.
- On startup, the REPL looks for .juliarc.jl and runs it. This is a good place to set up LOAD_PATH, and maybe load some modules that you always use. The file .juliarc.jl is in your home directory in Linux and in Windows at: `C:\\Users\\some_user\\AppData\\Roaming\\julia\\.juliarc.jl`
