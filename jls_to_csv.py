# jls to csv script
import numpy as np
from joulescope.data_recorder import DataReader

def jls_to_csv(fname: str, outname: str = None):
    """
    Converts the .jls datafile to a .csv.
    Note: the .csv is considerably larger than the .jls (for an idea, a .jls file of 
    size ~112 MB is ~799 MB!)
    """
    CHUNK_SIZE = 2**16

    if not fname.endswith('jls'):
        raise RuntimeError('file must be .jls type')

    if not bool(outname):
        outname = fname.replace('jls', 'csv')

    reader = DataReader()
    reader.open(fname)

    id_end = reader.sample_id_range[1]

    print(f'Converting {fname} to {outname}...')

    for i in range(0, id_end, CHUNK_SIZE):
        print('progress: {:.3} %\t\r'.format((i / id_end) * 100), end='')
        start = i
        end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

        # data[0] is current, data[1] is voltage
        data = reader.get_calibrated(start, end)

        ts = [i * 5e-7 for i in range(start, end)]
        data = np.asarray([ts, data[0], data[1]]).T

        with open(outname, 'ab') as outf:
            np.savetxt(outf,
                        data,
                        fmt='%2.7f %2.4e %2.4f',
                        delimiter=',',
                        header='time,current,voltage')

    print(f'Converted {fname} to {outname}')
    reader.close()

if __name__ == '__main__':
    import sys

    args = sys.argv
    if len(args) == 1:
        print('Error: Requires .jls file to convert')
        print('Usage: `python jls_to_csv.py <IN_FILE>.jls [<OUT_FILE>.csv]`')
        sys.exit()
    
    in_file = args[1]
    out_file = args[3] if len(args) >= 3 else None

    jls_to_csv(in_file, out_file)
