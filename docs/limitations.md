# DistillShield Limitations

- The dataset is synthetic. Behavioral separation is engineered, which makes metrics optimistic relative to real deployment conditions.
- The risk score is a research proxy, not a validated measurement of real exfiltration or true model distillation intent.
- The leakage proxy measures stylistic and structural exposure, not actual downstream distillation resistance.
- Infrastructure signals such as geographic implausibility, key rotation, and subnet clustering are synthetic or heuristic approximations.
- False positives are likely, especially for legitimate researchers, benchmark users, power users, and automated internal evaluation clients.
- The mock teacher response generator is deterministic and simplified; it does not represent the full variability of real LLM outputs.
- The grouped rules are hand-tuned for demonstrative clarity and will not generalise reliably to production traffic without real telemetry and validation.
- No authentication, abuse prevention, privacy controls, key management, audit logging hardening, or distributed deployment support is included.
- The project demonstrates feasibility of adaptive middleware mechanics, not proven security guarantees against model distillation.
