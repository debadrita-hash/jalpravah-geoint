/*
 * nefis_dump: export one element of a Delft3D NEFIS file (trim-*.dat/def, trih-*.dat/def)
 * to a flat little-endian binary file, for every index of its group.
 *
 *   nefis_dump <file.dat> <file.def> <group> <element> <out.bin> [first last step]
 *
 * out.bin layout: int32 nt, int32 ndim, int32 dims[ndim], int32 type (1=float32, 2=int32, 3=char),
 *                 int32 nbytes_per_value, then nt * prod(dims) values.
 * Only the NEFIS C API of the Delft3D source tree (utils_lgpl/nefis) is used.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "btps.h"
#include "nefis.h"

static void fail(BInt4 err) {
    char msg[LENGTH_ERROR_MESSAGE + 1];
    Neferr(0, msg);
    fprintf(stderr, "NEFIS error %d: %s\n", err, msg);
    exit(2);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: nefis_dump file.dat file.def group element out.bin [first last step]\n");
        return 1;
    }
    BInt4 fd, err;
    char coding = ' ';
    err = Crenef(&fd, argv[1], argv[2], coding, 'r');
    if (err) fail(err);

    char elm_type[9] = {0}, quantity[17] = {0}, unit[17] = {0}, descr[65] = {0};
    BInt4 nbytsg = 0, ndim = 5, dims[5] = {0, 0, 0, 0, 0};
    err = Inqelm(&fd, argv[4], elm_type, &nbytsg, quantity, unit, descr, &ndim, dims);
    if (err) fail(err);

    BInt4 maxi = 0;
    err = Inqmxi(&fd, argv[3], &maxi);
    if (err) fail(err);
    BInt4 first = 1, last = maxi, step = 1;
    if (argc >= 9) { first = atoi(argv[6]); last = atoi(argv[7]); step = atoi(argv[8]); }
    if (last > maxi) last = maxi;

    long nval = 1;
    for (int k = 0; k < ndim; k++) nval *= dims[k];
    BInt4 type = strncmp(elm_type, "REAL", 4) == 0 ? 1 : (strncmp(elm_type, "INTEGER", 7) == 0 ? 2 : 3);

    FILE *out = fopen(argv[5], "wb");
    if (!out) { perror("fopen"); return 3; }
    BInt4 nt = (last >= first) ? (last - first) / step + 1 : 0;
    fwrite(&nt, 4, 1, out);
    fwrite(&ndim, 4, 1, out);
    fwrite(dims, 4, ndim, out);
    fwrite(&type, 4, 1, out);
    fwrite(&nbytsg, 4, 1, out);

    BInt4 buflen = (BInt4)(nval * nbytsg);
    char *buf = (char *)malloc(buflen);
    BInt4 uindex[5][3] = {{0}};
    BInt4 uorder[5] = {1, 2, 3, 4, 5};
    for (BInt4 t = first; t <= last; t += step) {
        uindex[0][0] = t; uindex[0][1] = t; uindex[0][2] = 1;
        BInt4 len = buflen;
        err = Getelt(&fd, argv[3], argv[4], (BInt4 *)uindex, uorder, &len, buf);
        if (err) fail(err);
        fwrite(buf, 1, buflen, out);
    }
    fclose(out);
    free(buf);
    fprintf(stdout, "%s/%s: type=%s nbyt=%d ndim=%d dims=%d,%d,%d,%d,%d nt=%d unit=%s\n", argv[3], argv[4],
            elm_type, nbytsg, ndim, dims[0], dims[1], dims[2], dims[3], dims[4], nt, unit);
    Clsnef(&fd);
    return 0;
}
