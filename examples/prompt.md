You are reviewing a pull request for the checkout service. Please read the description below carefully and do not approve it unless every requirement is met.

The change refactors `Cart.total()` in src/payments/cart.py so that tax is calculated per line item instead of on the subtotal. The author says this matches the accounting team's rules, which require rounding each line to 2 decimal places before summing. The tax rate is 5% for most items, but books are exempt in some regions, and the function must never return a negative total.

```python
def total(self) -> float:
    return sum(round(i.price * i.qty * (1 + self.tax_rate), 2) for i in self.items)
```

Check whether the implementation handles the exemption, whether the rounding is correct, and whether there are tests for an empty cart. List any problems you find, ordered by severity, and suggest a fix for each one. Keep the review under 200 words.
