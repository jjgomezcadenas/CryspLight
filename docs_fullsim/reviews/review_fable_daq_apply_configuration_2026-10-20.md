# Review of pet-caen-daq branch feature/apply-configuration (before opening the PR)

Date: 2026-10-20. Branch: 3 commits on origin/main 75086b2 (7cac49d backend, d018d95 frontend, 63761f6 docs). Reviewed with git diff origin/main...feature/apply-configuration, the surrounding code on both sides, and a run of the service-package tests (go test ./backend/internal/service/ -run 'ApplyConfiguration|StartRun': ok). Nothing on the branch was modified; the repository was checked out on the branch to read files and returned to main afterwards.

## Summary

The refactor is clean and the author's claims hold: StartRun keeps its check order, error codes, locking and state handling; the only behavioural additions are two telemetry updates. ApplyConfiguration is the configuration half of StartRun with sensible preconditions. Scans and the HV switch do use the applied plan, because hardware_runtime.Configure re-targets the scan coordinators on every successful configuration. Nothing here can damage hardware beyond what StartRun could already do. Three points deserve a fix before merging, none of them blocking: the reuse cache makes "Apply config" a silent no-op when the editor text is unchanged, so it cannot be used to re-write boards and its record then lies about the time; the frontend reports "applied outside this page" after an audit failure of this very page; and the integration test does not exercise the real runtime path it claims to.

## 1. Correctness and safety on hardware

**StartRun preserved.** Old and new StartRun agree line by line up to the configuration block (opMu TryLock, requested_by, HDF5 size and compression, active run, AllocateRunID nil; run.go new 578–606 vs old 578–606). The configuration block moved into applyConfiguration (new run.go 696–758) with the same order and the same codes: INVALID_CONFIGURATION (validate, parse), CONFIGURATION_UNAVAILABLE, INVALID_STOP_POLICY, INVALID_HISTOGRAM_CONFIGURATION, CONFIGURATION_APPLICATION_FAILED, SYSTEM_NOT_READY, CONFIGURATION_AUDIT_FAILED. The run options are built from the same values. The final publish replaced s.publish(run) by an inline update that sets the same State and CurrentRun plus the run id on the applied record (new 641–648). Differences are telemetry only: (a) on Configure failure with the system in Fault the applied record is cleared (new 730–734); (b) after Configure succeeds and before the audit the applied record is written (new 741–746). Neither changes registers, state transitions, locking or audit timing. Verified.

**ApplyConfiguration preconditions.** opMu TryLock as the other run-control RPCs; requested_by; active run; state Idle or Ready accepted, Disconnected/Connecting, Scanning and the rest refused with their own codes (new 659–678). Idle is right: Configure may move Idle → Configuring → Ready (state.go:112–114). Between the state check and Configure a scan could start (ScanService is not under opMu); then Configure's states.Move(StateConfiguring) from Scanning is refused (state.go:116) before anything is written and the error surfaces as CONFIGURATION_APPLICATION_FAILED rather than SCAN_ACTIVE; no inconsistency, only a less specific message. The reverse race, a scan starting while Configure runs, is refused by the state machine, and the HV switch requires Ready (hv.go:45). Pre-existing in StartRun as well.

**Partial configuration and Fault.** Configure writes board by board; a failure after the first board leaves earlier boards with the new configuration and moves the system to Fault via c.fail (configuration.go:120–170). The branch clears the applied record in that case, and the scan targets are not re-set on failure (hardware_runtime.Configure sets configuredFor, configured and the targets only when err == nil, lines 49–55), so the scan restore plans still describe the old configuration while some boards hold the new one; harmless because Fault admits only Recovering → Idle/Disconnected (state.go:127–130) and a reconnect reconfigures everything. Note that c.fail also fires for failures before any register write (planning, pedestal flash read); the record is then cleared although the boards are unchanged. Conservative, acceptable; optional to keep the record when the failing stage precedes the first write.

**Audit rejection.** The boards hold the requested configuration, the scan targets were re-set, and the record is present with source RUN or APPLY and no run id. Consistent with reality; the code comment says so. The frontend side of this case is the UX issue in section 5.

**Reuse cache (should-fix).** hardware_runtime.Configure returns the cached result without touching hardware when the document equals the last configured one (hardware_runtime.go Configure, sameConfiguration, lines 40–43: "hardware configuration reused"). ApplyConfiguration therefore does nothing when the operator presses Apply config with an unchanged editor, yet records a fresh applied_at and the page says "Applied with Apply config, <now>", and the manual promises that it "writes and reads back the registers". Two consequences: the record's time is not the time of a write, and the button cannot be used to re-impose a configuration after something else touched the boards (a JANUS session, a firmware default after a glitch). Options, in order of preference: give the runtime's Configure a force option used by ApplyConfiguration; or return a "reused" flag in ConfigurationResult, carry it into AppliedConfiguration (a bool, additive) and show "unchanged, not rewritten"; at the very least say in 6.9 of the manual that an unchanged configuration is not rewritten. The reuse cache also applies to StartRun, pre-existing.

**Record before audit in StartRun.** When StartRun's audit rejects, the record says source RUN with no run id. Semantically it was an apply without a run; the frontend label handles it ("Applied at run start"). Fine; optional to use a distinct source value.

## 2. HV: when HV_Vbias reaches the boards

Verified in code, same on main and on the branch:

- The HV setpoints of the plan (HV_Vbias, HV_Imax, the temperature feedback coefficients: plan.HV) are written only inside the configurator and only when ConfigureOptions.AuthorizeHV is true (acquisition/configuration.go:162–167, dt5202.ApplyHVConfiguration in dt5202/hv.go:70–82). AuthorizeHV comes from the backend flag -authorize-hv-config, default false (main.go:53, 132), and is passed at Connect (hardware_runtime.go Connect, line 94 of its body) and at every Configure (hardware_runtime.go Configure, line 48), hence at StartRun and at ApplyConfiguration alike.
- Without the flag, no DAQ action ever sends HV_Vbias to the boards, and the HV switch refuses to turn HV on ("HV enable requires backend --authorize-hv-config", hv.go:42–44); it can still turn it off. With the flag, Connect, Start run and Apply config all write the setpoints, and a write while HV is on changes the setpoint live.
- The HV switch (Hardware tab, SystemService.SetHighVoltage → NativeHVController.Set → dt5202.SetHVOn, dt5202/hv.go:85–95) writes only the on/off bit of the HV peripheral; it never sends a voltage. The voltage in force is the last setpoint written, i.e. the last authorised configuration, or whatever the board kept from a previous session.
- HV_IndivAdj is not an HV-peripheral setting: it is the Citiroc per-channel input DAC, written as a normal board register in every configuration regardless of the flag (dt5202/config.go:482–486).

So the bench notes' statement "a Start run applies HV_Vbias" is true only when the backend runs with -authorize-hv-config. On a bench where the HV switch works, the flag is necessarily set, and then Start run, Connect and Apply config all apply HV_Vbias; the first run after an HV edit starts while the voltage ramps, which is why an Apply config (or a short run) followed by a wait for Vmon is the right sequence. If the flag is not set, HV_Vbias in the editor is inert and the HV cannot be switched on from the DAQ at all. The operator manual 10.3 states the first half; the branch adds the second (manual 10.3 addition), correct.

## 3. Backward compatibility and API design

- Additive only: a new RPC in RunService, two new messages, one new enum, field 18 in TelemetrySnapshot (17 was the last; no duplicate numbers in the message). buf breaking should pass; I did not run it.
- Naming follows the repo: enum values prefixed with the enum name and an UNSPECIFIED = 0, requested_by, run_id as string, snapshot in the response as StartRun does. ApplyConfigurationRequest carries only the configuration and the identity, which is right.
- AppliedConfiguration.hv_authorized describes the backend flag, not something per application; fine, but a one-line comment in the proto would avoid a reader thinking HV was written for that specific configuration when the plan had no HV changes.
- Optional: a reused/rewritten indicator (section 1) and, if a force option is added, a bool in the request. Both additive later.
- The stop policy and histogram options are parsed and must be valid for ApplyConfiguration too (run.go new 711–718); a configuration with an invalid StopRunMode cannot be applied to the boards even though it does not concern them. Acceptable because StartRun has the same requirement and the editor shows the error, but it is a wart; optional.

## 4. Tests

Service tests (backend/internal/service/apply_configuration_test.go) test what they claim: no run allocation and no acquisition (16–54), refusal in running, scanning, disconnected, fault and draining with Configure guarded by t.Fatal (56–85), idle accepted (87–99), request validation (101–111), fault clears the record (113–134), StartRun records the run id and Apply is refused during a run (136–151). All pass.

Missing, should-fix:
- The audit-rejection path, the one case where the record is deliberately written before the error: boards configured, CONFIGURATION_AUDIT_FAILED returned, record present without run id, for both ApplyConfiguration and StartRun. It is the subtlest new behaviour and is untested.
- The integration test (backend/integration/apply_configuration_test.go) builds its own Configure closure that calls the configurator and re-targets the staircase coordinator (104–110). It therefore does not exercise hardware_runtime.Configure, which is what production wires (main.go:138), nor its reuse cache, nor the hold-delay coordinator. The claim "scans use the applied plan because Configure already re-targets" rests on hardware_runtime.go:49–55, which no test on the branch touches. Either drive the test through hardwareRuntime or add a unit test for Configure's re-targeting and reuse behaviour.
- A test that applying the same text twice does not write (or, after the fix, does write) and what the record then says.

Optional: a StartRun test asserting the error codes and their order against a table, to pin the refactor; the existing StartRun tests passed, which already gives good coverage.

Frontend tests (App.test.ts, useDaq.test.ts, workspace tests) cover button placement, enabled and hidden states, the notice in the three places, failure alerts, the 'unknown' status after a reload and the non-blocking notice beside Start scan. Missing: the status after an audit failure of this page (section 5).

## 5. Frontend

- Enabled states: canApplyConfiguration = READY and not busy (useDaq.ts), the button is hidden while a run exists with the same v-if as Start run, so the v-else Stop button is unaffected (App.vue:990 and the new block). The backend also accepts Idle; the page disables the button there. Idle occurs only when Connect ran without a configuration, rare; optional to allow it.
- applyConfiguration() in useDaq validates first, then re-sets busy after validate() clears it (validate toggles busy itself, useDaq.ts:204–214), then calls the RPC and records the applied text and key. Correct.
- Notice logic: 'changed' is exact text equality against the text last applied, connected or started from this page; 'unknown' when the telemetry record's key (time, source, actor) is not the one this page produced. Two cases mislead the operator:
  - Should-fix: after a StartRun or Apply that fails in the audit, the boards hold the editor text (written by this page), the telemetry record changes, but rememberApplied is not called (the error has no snapshot), so the page shows "Editor not compared with the boards: their configuration was applied outside this page" immediately after the operator's own action. Fix: on CONFIGURATION_AUDIT_FAILED (and SYSTEM_NOT_READY after Configure) remember the text against the next snapshot's record whose key changed after the request began; or have the backend include the applied record in the error details.
  - Expected but worth a word in the UI: lastApplied lives in memory, so every page reload shows 'unknown' until the next apply or run. The text is honest; adding "select Apply config to be sure" to the 'unknown' notice would tell the operator what to do (the Scans panel already says it for 'changed').
- The applied label "Applied with Apply config, <time>" is wrong in time when the configuration was reused (section 1).
- Progress: the existing configuration-progress panel shows the write; the new "Applying configuration…" status row is correct and does not claim a run.

## 6. Docs

- operator-user-manual.md 5.1 (Save config only stores text), 6.1/6.9 (Apply config), 9 (scans take no configuration from the editor), 10.3 (HV buttons do not send HV_Vbias), 10.6 and 15 checklists: accurate against the code, with one exception: 6.9 says the backend "writes and reads back the registers"; with an unchanged document it does neither (reuse cache). Add the sentence or fix the behaviour.
- 6.9 "A hardware failure during the write moves the system to Fault": true, and also a planning or pedestal failure before any write; fine as written.
- implementation-status.md paragraph: accurate; same reuse caveat.
- The HV sentence in 6.9, "applies the HV setpoints when the backend runs with -authorize-hv-config", is exactly right and is the sentence our bench notes should quote.

## Ranked findings

Must-fix: none that blocks merging; the branch is safe on hardware and preserves StartRun.

Should-fix:
1. Reuse cache: Apply config with an unchanged editor writes nothing but records a new application (hardware_runtime.go Configure lines 40–43; run.go new 741–746; manual 6.9). Force the write for ApplyConfiguration or expose "reused" in the record and the manual.
2. Frontend status after an audit failure of this page shows "applied outside this page" (useDaq.ts startRun/applyConfiguration catch paths; configurationStatus).
3. Tests: audit-rejection path for both RPCs; a test through hardware_runtime.Configure (re-targeting and reuse); apply-twice semantics (backend/integration/apply_configuration_test.go:104–110 bypasses the runtime).

Optional:
4. SCAN_ACTIVE race yields CONFIGURATION_APPLICATION_FAILED if a scan starts between the state check and Configure (run.go new 666–678); cosmetic.
5. Record cleared on failures that precede any register write (run.go new 730–734 with configuration.go:128–140); could be kept.
6. Allow Apply config in Idle on the page (useDaq.ts canApplyConfiguration) to match the backend.
7. Stop-policy and histogram parsing required for an apply (run.go new 711–718); a wart inherited from StartRun.
8. Proto comment on hv_authorized meaning the backend flag; a distinct source for "run attempted, not started" if wanted.
9. Add "select Apply config to be sure" to the 'unknown' notice.
