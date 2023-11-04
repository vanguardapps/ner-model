import utils

filepath = input('Enter the filepath for the dataset to split: ')
batch_name = input('Enter the base name for the generated batches: ')
lines_per_batch = int(input('Enter the number of lins per batch: '))
repeat_first_line = input('Do you want to repeat the first line for every batch? (y/n): ')

batch = []
number_of_batches = 1
first_line = ''

def write_batch():
     global number_of_batches
     global batch
     global repeat_first_line
     global first_line
     to_write = [first_line] + batch if repeat_first_line.lower() == 'y' else batch
     with open(batch_name + "-" + ("%s" % number_of_batches).zfill(5), "w", encoding="utf-8") as f:
        f.write(''.join(to_write))
        batch = []
        number_of_batches += 1

with open(utils.relative_path(filepath), "r", encoding="ISO-8859-1") as f:
    for index, line in enumerate(f):
        if index == 0 and repeat_first_line.lower() == 'y':
            first_line = line
        else:
            batch.append(line)
        if ((index + 1) % lines_per_batch == 0):
            write_batch()
    
if len(batch) > 0:
    write_batch()

print(f"Wrote total number of batches: {number_of_batches}")
print(F"Linse per batch: {lines_per_batch}")
            

        