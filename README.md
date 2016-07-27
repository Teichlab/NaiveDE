This package implements the simplest form of a Differential Expression test.
It just fits two linear models, assuming Normally distributed residuals, to
each gene. These are assumed to be nested, and a likelihood ratio test is
then performed to compare the models.

The test might have low power and give noisy results, but it shouldn't be
biased. With many (i.e. hundreds of) samples it should work all right.

This can serve as a baseline comparison with more sophisticated tests.

## Faux fold changes

The package also has methods for creating input fold-change controlled
fake conditions in data using ERCC spike-ins. This performs systematic
renaming of spike-ins in randomized conditions. To avoid creating unrealistic
levels of fold change, input concentration is used to limit possible renaming.

This assumes expression measures used are comparable between different
sequences. Thus these should optimally account for length and other sequence
features before creating faux fold changes.
