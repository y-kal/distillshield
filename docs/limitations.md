# DistillShield Limitations

- The dataset is synthetic. Behavioral separation is engineered, which makes metrics optimistic relative to real deployment conditions.
- The risk score is a research proxy, not a validated measurement of real exfiltration or true model distillation intent.
- The leakage proxy measures stylistic and structural exposure, not actual downstream distillation resistance.
- Infrastructure signals such as geographic implausibility, key rotation, and subnet clustering are synthetic or heuristic approximations.
- False positives are likely, especially for legitimate researchers, benchmark users, power users, and automated internal evaluation clients.
- The mock teacher response generator is deterministic and simplified; it does not represent the full variability of real LLM outputs.
- The baseline models are trained only on local synthetic features and will not generalize reliably to production traffic without real validation data.
- No authentication, abuse prevention, privacy controls, key management, audit logging hardening, or distributed deployment support is included.
- Optional sequence and graph modules are interface-complete but not validated as useful defenses in this prototype.
- The project demonstrates feasibility of adaptive middleware mechanics, not proven security guarantees against model distillation.
