import struct

def read_binary_file(filename, block_size, num_blocks):
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        data = []
        # Read the blocks one by one
        for _ in range(num_blocks):
            # Read a block of data
            block_data = file.read(block_size)
            if not block_data:
                break  # If no more data, break
            # Unpack the block data into a list of integers (unsigned 8-bit)
            block_values = list(struct.unpack(f'{block_size}B', block_data))
            data.append(block_values)
    return data

def print_binary_data(data):
    for i, block in enumerate(data):
        print(f"Block {i+1}:")
        for value in block:
            # Print each value in decimal (uint8_t) and hexadecimal (0xXX) format
            print(f"{value} (0x{value:02X})", end=" ")
        print()  # Newline after each block

if __name__ == "__main__":
    # Specify the filename, block size, and number of blocks
    filename = "LDPC_ul_encoded_1024_ant1.bin"
    block_size = 363  # Set this to the value of `ul_cb_bytes` from your C++ code
    num_blocks = 14    # Replace with the actual number of blocks

    # Read the binary file
    binary_data = read_binary_file(filename, block_size, num_blocks)

    # Print out the values
    print_binary_data(binary_data)

