🚀 Method 2 (IMPORTANT): Direct light sampling (Next Event Estimation)

This is how real path tracers get clean results.

Idea

At each surface hit:

Explicitly sample the light

Shoot a shadow ray toward it

Add its contribution directly

This avoids waiting for random bounces to find the light.