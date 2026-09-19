# IEM EarPrint Engine — Dynamic

A deterministic Python implementation of the EarPrint workflow designed so that the **data directory is the source of truth**.

## Dynamic IEM discovery

Every:

```text
input/preferred/*.txt
```

is automatically discovered and treated as **one independent IEM vote**.

The calculation does not depend on a fixed number of IEMs.

For example:

```text
Pudding.txt
Ceramics_Ultra.txt
Kato.txt
Origin.txt
```

means four independent IEM votes.

Adding another valid preferred-response file automatically adds one independent vote on the next build. Removing one removes that vote.

**One IEM = one vote.**

Repeated measurements of the same IEM must not become additional independent votes.

## Dynamic target discovery

Every:

```text
input/targets/*.txt
```

is automatically discovered and processed independently.