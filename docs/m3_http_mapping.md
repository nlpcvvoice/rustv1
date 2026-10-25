# M3 HTTP mapping (consultation #22)

HTTP mapping

## Accounting correction

The following numbering is an explicit enumeration of the **46 successful comparison requests recoverable from the supplied driver**:

| Category | Requests |
|---|---:|
| Singleton/reference | 22 |
| Multi-input chunks | 6 |
| Short + 1024 | 1 |
| Reverse and shuffle | 2 |
| Duplicates | 1 |
| Repeated shape | 3 |
| Alternating lengths | 6 |
| Recovery singletons | 5 |
| **Total** | **46** |

The five invalid requests and two poison checks are additional tests.

### Fixture notation and limits

- `P[i]`: exact token IDs from `results/prompts.json[i]`.
- `s`: first shortest sample, as selected by the driver; here sample **0**.
- `l`: first sample whose parsed length is exactly 1024. Compute this from the file, rather than inferring it from motif descriptions.
- `T[i]`: a text fixture **verified through the production HTTP tokenizer path** to produce exactly `P[i]`.
- `U[i]`: an explicitly approved served-text substitute when exact representation is unavailable. Its expected IDs must come from an independently verified fixture descriptor, **not** from `P[i]`.
- `R`: `[21,20,...,0]`.
- `Q`: the driver’s exact wrapping-u64 Fisher–Yates permutation.

**No tokenizer files, verified HTTP fixture texts, production tokenizer configuration, or #19 Part 1 text are attached.** Therefore the table below is complete as a **request mapping**, but exact `T[i]` availability and the concrete substitute texts cannot honestly be marked verified. Decoding a token list alone is not a round-trip proof.

For every row using `T[I]`, the fixture-text source is the to-be-verified HTTP fixture descriptor; the expected-ID source is `P[I]`. If any member is replaced by `U[i]`, that member’s HTTP golden must be generated for `U[i]`’s actual IDs, and the original token-ID case remains in-process-only.

| # | Driver request | HTTP case(s) / ordered fixture source | Expected token IDs | Split behavior |
|---:|---|---|---|---|
| 1 | `singleton_reference(0)` | `singleton-00`: `[T[0]]` | `[P[0]]` | None |
| 2 | `singleton_reference(1)` | `singleton-01`: `[T[1]]` | `[P[1]]` | None |
| 3 | `singleton_reference(2)` | `singleton-02`: `[T[2]]` | `[P[2]]` | None |
| 4 | `singleton_reference(3)` | `singleton-03`: `[T[3]]` | `[P[3]]` | None |
| 5 | `singleton_reference(4)` | `singleton-04`: `[T[4]]` | `[P[4]]` | None |
| 6 | `singleton_reference(5)` | `singleton-05`: `[T[5]]` | `[P[5]]` | None; retain stress sample |
| 7 | `singleton_reference(6)` | `singleton-06`: `[T[6]]` | `[P[6]]` | None |
| 8 | `singleton_reference(7)` | `singleton-07`: `[T[7]]` | `[P[7]]` | None |
| 9 | `singleton_reference(8)` | `singleton-08`: `[T[8]]` | `[P[8]]` | None |
| 10 | `singleton_reference(9)` | `singleton-09`: `[T[9]]` | `[P[9]]` | None |
| 11 | `singleton_reference(10)` | `singleton-10`: `[T[10]]` | `[P[10]]` | None |
| 12 | `singleton_reference(11)` | `singleton-11`: `[T[11]]` | `[P[11]]` | None |
| 13 | `singleton_reference(12)` | `singleton-12`: `[T[12]]` | `[P[12]]` | None |
| 14 | `singleton_reference(13)` | `singleton-13`: `[T[13]]` | `[P[13]]` | None |
| 15 | `singleton_reference(14)` | `singleton-14`: `[T[14]]` | `[P[14]]` | None |
| 16 | `singleton_reference(15)` | `singleton-15`: `[T[15]]` | `[P[15]]` | None |
| 17 | `singleton_reference(16)` | `singleton-16`: `[T[16]]` | `[P[16]]` | None |
| 18 | `singleton_reference(17)` | `singleton-17`: `[T[17]]` | `[P[17]]` | None |
| 19 | `singleton_reference(18)` | `singleton-18`: `[T[18]]` | `[P[18]]` | None |
| 20 | `singleton_reference(19)` | `singleton-19`: `[T[19]]` | `[P[19]]` | None |
| 21 | `singleton_reference(20)` | `singleton-20`: `[T[20]]` | `[P[20]]` | None |
| 22 | `singleton_reference(21)` | `singleton-21`: `[T[21]]` | `[P[21]]` | None |
| 23 | `multi_input([0,1,2,3])` | `multi-0`: `T[0:4]` | `P[0:4]` | None |
| 24 | `multi_input([4,5,6,7])` | `multi-1`: `T[4:8]` | `P[4:8]` | None |
| 25 | `multi_input([8,9,10,11])` | `multi-2`: `T[8:12]` | `P[8:12]` | None |
| 26 | `multi_input([12,13,14,15])` | `multi-3`: `T[12:16]` | `P[12:16]` | None |
| 27 | `multi_input([16,17,18,19])` | `multi-4`: `T[16:20]` | `P[16:20]` | None |
| 28 | `multi_input([20,21])` | `multi-5`: `T[20:22]` | `P[20:22]` | None |
| 29 | `multi_input_short_1024` | `short-long`: `[T[s],T[l]]` | `[P[s],P[l]]` | None |
| 30 | `reversed` | `reverse-a/b/c`: `T[R]` | `P[R]` | Contiguous ordered slices 8/8/6 |
| 31 | `deterministic_shuffle` | `shuffle-a/b/c`: `T[Q]` | `P[Q]` | Contiguous ordered slices 8/8/6 |
| 32 | `duplicates` | `duplicates`: `[T[s],T[l],T[s],T[l]]` | `[P[s],P[l],P[s],P[l]]` | None |
| 33 | `repeat_identical_shape_0` | `repeat-0`: `[T[s],T[l]]` | `[P[s],P[l]]` | None |
| 34 | `repeat_identical_shape_1` | `repeat-1`: `[T[s],T[l]]` | `[P[s],P[l]]` | None |
| 35 | `repeat_identical_shape_2` | `repeat-2`: `[T[s],T[l]]` | `[P[s],P[l]]` | None |
| 36 | `alternating_length_0` | `alternate-0`: `[T[s]]` | `[P[s]]` | None |
| 37 | `alternating_length_1` | `alternate-1`: `[T[l]]` | `[P[l]]` | None |
| 38 | `alternating_length_2` | `alternate-2`: `[T[s]]` | `[P[s]]` | None |
| 39 | `alternating_length_3` | `alternate-3`: `[T[l]]` | `[P[l]]` | None |
| 40 | `alternating_length_4` | `alternate-4`: `[T[s]]` | `[P[s]]` | None |
| 41 | `alternating_length_5` | `alternate-5`: `[T[l]]` | `[P[l]]` | None |
| 42 | `recovery_after_empty_batch` | `recover-empty-batch`: `[T[s]]` | `[P[s]]` | Immediately after corresponding rejection |
| 43 | `recovery_after_empty_sequence` | `recover-empty-sequence`: `[T[s]]` | `[P[s]]` | Immediately after corresponding rejection |
| 44 | `recovery_after_pad_token` | `recover-pad`: `[T[s]]` | `[P[s]]` | After exact text case if representable; otherwise after documented substitute rejection |
| 45 | `recovery_after_over_length` | `recover-overlength`: `[T[s]]` | `[P[s]]` | Immediately after corresponding rejection |
| 46 | `recovery_after_out_of_vocabulary` | `recover-oov-substitute`: `[T[s]]` | `[P[s]]` | After served request-type rejection; not an OOV-text test |

These become **50 positive HTTP requests** after splitting reverse and shuffle, before additional served-only cases or budget-driven splits. Splitting does not establish the behavior of a single 22-input HTTP request; that request is outside the stated eight-string limit.

Exact shuffle construction:

```python
q = list(range(22))
state = 0x5256314D32534855
for i in range(21, 0, -1):
    state = (
        state * 6364136223846793005 + 1442695040888963407
    ) & ((1 << 64) - 1)
    j = state % (i + 1)
    q[i], q[j] = q[j], q[i]
```

### Invalid and operational cases

| Original case | HTTP treatment | In-process-only boundary / substitute |
|---|---|---|
| Empty logical batch | Submit an empty text array | Direct analogue if the HTTP schema permits expressing the array |
| Valid prefix + empty sequence | Submit `[T[s], ""]`; verify rejection before any forward | Empty string must not silently become a nonempty special-token sequence; HTTP validation and token validation are distinct |
| Valid prefix + `[11]` | Use a verified text encoding exactly to `[11]`, only if available | Otherwise exact pad-ID test remains in-process-only; use `[T[s], ""]` as a separately named served prevalidation test, not a pad-token equivalent |
| Valid prefix + 1025 repeated IDs | Use verified text producing more than 1024 IDs without truncation | Exact repeated-ID fixture may remain in-process-only; served overlength fixture tests the boundary, not identical token contents |
| Valid prefix + `[vocab_size]` | No legitimate tokenizer output can contain that OOV ID | In-process-only; submit a non-string array element as a separately named HTTP schema-rejection substitute |
| Operational policy-drift injection | No production HTTP mutation endpoint should be added | Original injection remains in-process-only; served counterpart requires a test-only fault-injection harness |
| Request after poisoning | Test through the served harness after injected operational failure | An ordinary valid request to a healthy server is not a substitute for poison-guard coverage |

For invalid requests, HTTP status alone is insufficient: preserve evidence that no physical forward was added and that the next valid request succeeds. For operational failure, preserve sticky refusal evidence.

**Remaining mapping work:** materialize and verify the `T/U` descriptor, including an actual 1024-token served fixture. The attached IDs alone do not establish which individual corpus rows round-trip through the production tokenizer.
