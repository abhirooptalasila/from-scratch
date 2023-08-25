# baby86

A minimal x86 bootloader.

A bootloader is the absolute first time to run when a computer boots up (handled by the BIOS on an x86 PC). Here's the immediate steps that happen after you turn on your PC:

- the processor looks at the address 0xFFFFFFF0 for the BIOS code. It is a piece of ROM
- BIOS then POSTs and searches for all available boot media (boot sector should end in 0x55AA)
- If the BIOS finds some bootable drive, it loads the first 512 bytes into address 0x007C00
- Transfers control to processor with a jump instruction to the above address

- I use NASM and the Bochs x86 IBM-PC compatible emulator

Run:
`$ bochs -f bochsrc.txt`

[comment]: <> (https://www.joe-bergeron.com/posts/Writing%20a%20Tiny%20x86%20Bootloader/)