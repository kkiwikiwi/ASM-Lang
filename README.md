## Installation
`Invoke-WebRequest -Uri "https://github.com/python-processing-unit/ASM-Lang/archive/refs/heads/main.zip" -OutFile "path\to\download\ASM-Lang.zip"`<br>
`Expand-Archive -Path "path\to\download\ASM-Lang.zip" -DestinationPath "path\to\extract\ASM-Lang"`<br>
`$old = [Environment]::GetEnvironmentVariable('Path','User')`<br>
`if(-not $old.Split(';') -contains 'path\to\extract\ASM-Lang'){ [Environment]::SetEnvironmentVariable('Path',$old + ';path\to\extract\ASM-Lang','User') }`<br>
`Remove-Item -Path "path\to\download\ASM-Lang.zip"`<br>

## Crypto extension
The crypto extension (RSA, AES-GCM, AES-CBC) requires the Python package `cryptography`.

Install:
`python -m pip install cryptography`

Use:
`IMPORT(crypto)`

Notes:
- AES-GCM requires a 12-byte IV (nonce) and callers must ensure IV uniqueness per key.
- RSA_KEYGEN requires at least 2048 bits.
- AES-CBC does not provide authenticity; prefer AES-GCM for encrypt-then-auth.
- `crypto.ZERO_BYTES(TNS)` overwrites a byte tensor in-place (best-effort; STR/PEM keys cannot be zeroed).

## Tests
Run the crypto + GC tests:
`python asm-lang.py test_crypto_gc.asmln`

Run the function value tests:
`python asm-lang.py test_functions.asmln`

## Function values
Functions are first-class values with the `FN` type. You can assign them to variables, pass them to other functions, and return them.

Example:
```
FUNC INC(INT:x):INT{
    RETURN(ADD(x,1))
}

FUNC APPLY(FN: op, INT: value):INT{
    RETURN(op(value))
}

FN: fn = INC
INT: out = APPLY(fn, 1)
PRINT(out)
```
