Some random tips:
- The Julia binaries may be very old. It is much better to compile Julia from source (available in GitHub).
- The manual usually lags behind the code.
- Julia does not have a mechanism to clear variables, constants, types , functions. If you are using Julia from the REPL (the console), it is easiest to just restart the REPL when you want to clear stuff.

Setting up LOAD_PATH
- On startup, the REPL looks for `.juliarc.jl` and runs it. This is a good place to set up `LOAD_PATH`, and maybe load some modules that you always use. The file `.juliarc.jl` is in your home directory in Linux and in Windows at: `C:\\Users\\some_user\\AppData\\Roaming\\julia\\.juliarc.jl`.
- My `.juliarc.jl` contains:

  `appendloadpath(path) = (push!(LOAD_PATH,path);println("--> appending \"$(path)\") to LOAD_PATH"))`
  `appendloadpath("my source path")`
  
  `using GenUtils`


Organising your source code (Just a suggestion. Many other schemes are possible):
- Use modules. Put everything you do in modules. 
- It helps to declare one module per file. Put the  module `MyModule` in `MyModule.jl` somewhere that is accessible in `LOAD_PATH`. If you do this, you can load your module from anywhere with the statement `using  MyModule`. Otherwise, you have to use `require("some_path\MyModule.jl")`, followed by `using MyModule`.
- If your module becomes too long for one file, in that file put `include("mymodule/some_name.jl")`. The include is relative to the location of MyModule.jl. So you need to put the other source files in a subdirectory.
- You can also use submodules. See the manual and experiment with how this works.



Learning what Julia does and how:
- Have the Julia source handy so you can seach or grep into it. Usually you will want to search in the Julia sources in your Julia installation at: `.../share/julia/base`.
- The macro `@which` is your friend.  E.g., if you want to know what happens when you do calculate `A'*B`, do `@which A'*B` and go to the source file indicated.
- The macro `@elapsed` helps to time your calculations.
