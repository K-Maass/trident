# Trading mode

Trident defaults to paper mode. The release demo uses the simulated broker and requires no brokerage account. The paper adapter needs separate paper credentials.

The broker adapter also has a guarded live path. `TRIDENT_ENV=live` requires both an explicitly armed `config/live.yaml` and separate live credentials. The checked-in config is disarmed. Treat live operation as a separate validation effort covering order recovery, reconciliation, and risk controls.
