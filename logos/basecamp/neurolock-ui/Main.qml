import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// Neurolock PoP tab. Three actions (Enrol, Verify, Delegate to agent) plus a
// status view. Every action goes: this tab -> neurolock_pop core module ->
// the local Neurolock service (logos/pop_service.py) -> veins / pulse / EEG /
// chain. The log shows which layer made each decision.
Item {
    id: root
    width: 760
    height: 680

    property var st: ({})
    property bool busy: false

    // ── bridge to the core module ───────────────────────────────────────
    function call(method, args) {
        if (typeof logos === "undefined" || !logos.callModule) return null
        let raw = logos.callModule("neurolock_pop", method, args)
        try {
            let v = (typeof raw === "string") ? JSON.parse(raw) : raw     // IPC wraps strings once
            return (typeof v === "string") ? JSON.parse(v) : v
        } catch (e) { return { error: "bad reply: " + raw } }
    }
    function refresh() {
        let s = call("state", [])
        if (s) root.st = s
        let jobs = (s && s.jobs) ? s.jobs : []
        root.busy = jobs.length > 0 && jobs[jobs.length - 1].status === "running"
    }
    function act(action) {
        let r = call("start", [action, appField.text])
        if (r && r.error) logView.extra = r.error
        refresh()
    }
    Timer { interval: 1000; running: true; repeat: true; onTriggered: refresh() }
    Component.onCompleted: refresh()

    // ── layout ──────────────────────────────────────────────────────────
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 12

        RowLayout {
            Text { text: "Neurolock PoP"; font.pixelSize: 24; font.weight: Font.DemiBold }
            Item { Layout.fillWidth: true }
            Text {
                text: st.error ? "service offline" : ("ledger: " + (st.ledger || "?") +
                      (st.program_id ? "  ·  program " + st.program_id.substring(0, 8) + "…" : ""))
                color: st.error ? "#b3261e" : "#555"
            }
        }
        Text {
            text: "Veins prove who you are · pulse + EEG flicker prove you are here · the chain holds only a commitment. " +
                  "Pulse and EEG responses are SIMULATED in this demo."
            wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#666"; font.pixelSize: 12
        }

        // status
        RowLayout {
            spacing: 18
            Text { text: (st.enrolled ? "● enrolled" : "○ not enrolled"); color: st.enrolled ? "#1b7f3b" : "#888" }
            Text { text: (st.verified ? "● verified this session" : "○ not verified"); color: st.verified ? "#1b7f3b" : "#888" }
            Text { text: st.person || ""; color: "#666" }
        }

        // actions
        RowLayout {
            spacing: 8
            Button { text: "Enrol"; enabled: !busy && !st.enrolled; onClicked: act("enrol") }
            Button { text: "Verify"; enabled: !busy && st.enrolled; onClicked: act("verify") }
            Button { text: "Verify + consent"; enabled: !busy && st.enrolled; onClicked: act("consent") }
            TextField { id: appField; text: "fieldstation-vote"; Layout.preferredWidth: 170 }
            Button { text: "Delegate to agent"; enabled: !busy && st.verified; onClicked: act("delegate") }
            Button { text: "Revoke"; enabled: !busy && st.verified; onClicked: act("revoke") }
        }
        RowLayout {
            spacing: 8
            Text { text: "Attacks:"; color: "#666" }
            Button { text: "Impostor (Bob)"; enabled: !busy && st.enrolled; onClicked: act("impostor") }
            Button { text: "Replayed EEG"; enabled: !busy && st.enrolled; onClicked: act("replay") }
            BusyIndicator { running: busy; implicitHeight: 28; implicitWidth: 28 }
        }

        // delegations
        Text { text: "Delegations"; font.weight: Font.DemiBold }
        Repeater {
            model: st.delegations ? Object.keys(st.delegations) : []
            delegate: Text {
                property var d: st.delegations[modelData]
                text: modelData + "  →  agent " + (d.agent || "?") + "   " + (d.active ? "ACTIVE" : "inactive") + "   (" + d.id + ")"
                color: d.active ? "#1b7f3b" : "#888"
            }
        }

        // log: one line per layer decision
        Text { text: "What each layer decided"; font.weight: Font.DemiBold }
        Rectangle {
            Layout.fillWidth: true; Layout.fillHeight: true
            color: "#111"; radius: 6
            ScrollView {
                id: logView
                property string extra: ""
                anchors.fill: parent; anchors.margins: 8
                TextArea {
                    readOnly: true; wrapMode: Text.WrapAnywhere
                    color: "#e6e6e6"; font.family: "monospace"; font.pixelSize: 12
                    background: null
                    text: (st.log ? st.log.map(function (l) { return l.t + "  " + l.line }).join("\n") : "") +
                          (logView.extra ? "\n" + logView.extra : "")
                }
            }
        }
    }
}
