# gpu-count-smp
---

# Build

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
cmake --build build --parallel $(nproc) --config Release
```

# Run

## For $n \le 4$

```bash
./build/bruteforce
```

<details>

<summary>Sample output:</summary>

```
Calculating n=1 matchCountSize=1 (n!)^n=1
Time: 0ms
1
Max count: 1
Sum: 1 == (n!)^{2n}

Calculating n=2 matchCountSize=2 (n!)^n=4
Time: 0ms
14,2
Max count: 2
Sum: 16 == (n!)^{2n}

Calculating n=3 matchCountSize=3 (n!)^n=216
Time: 0ms
34080,11484,1092
Max count: 3
Sum: 46656 == (n!)^{2n}

Calculating n=4 matchCountSize=10 (n!)^n=331776
Time: 1785ms
65867261184,35927285472,7303612896,861578352,111479616,3478608,581472,36432,0,144
Max count: 10
Sum: 110075314176 == (n!)^{2n}
```

</details>

## For $n = 5$

**Fixed men preference:**

```bash
./build/random_menPrefs
```

<details>

<summary>Sample output:</summary>

```
Calculating n=5 matchCountSize=16 nrInstances=24883200000 menPrefId=22860193776
4,0,1,2,3
2,1,0,3,4
1,2,3,4,0
1,0,4,3,2
4,2,1,0,3
  ████████████████████████████████████████▏ 100.0% [  25/  25 | 3.5 Hz | 7s<0s]   
Time: 7075ms
62830760,73791600,41312200,17613950,7810630,2384670,972860,429225,137240,57250,15290,3540,380,390,0,15
Max count: 16

Calculating n=5 ...
```

</details>

**Validating:**

```bash
./build/find_womenPref_with_targetCount 22860193776 16 5496375240
```

<details>

<summary>Sample output:</summary>

```
40
menPrefId: 22860193776 targetCount: 16
4,0,1,2,3
2,1,0,3,4
1,2,3,4,0
1,0,4,3,2
4,2,1,0,3

womenPrefId: 5496375240
0,1,2,3,4
3,2,1,4,0
3,4,1,0,2
2,3,0,1,4
1,0,3,2,4

count: 16 == targetCount
```

</details>
