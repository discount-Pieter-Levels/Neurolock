import QtQuick
import QtQuick.Layouts
import "nl_data.js" as Data
import "sha256.js" as Sha

// NeuroLock station demo, native Basecamp tab (port of demo_web/index.html).
// Inputs are recorded samples dragged into place; every number shown comes from
// the real pipeline (export_demo_data.py -> build_assets.py -> nl_data.js).
Rectangle {
    id: root
    anchors.fill: parent
    color: c.bg

    readonly property var nl: Data.NL
    readonly property var pm: nl.params
    property int step: 0
    property string who: ""
    property alias testCon: con          // test hooks (offscreen driver)
    property alias testConPage: conPage

    QtObject {
        id: c
        readonly property color bg: "#0b0b0b"
        readonly property color line: "#242424"
        readonly property color line2: "#3a3a3a"
        readonly property color ink: "#ededed"
        readonly property color dim: "#8c8c8c"
        readonly property color faint: "#555555"
        readonly property color ok: "#5cb88a"
        readonly property color bad: "#e5534b"
    }

    // ── helpers ────────────────────────────────────────────────────────
    function after(ms, fn) {
        var t = Qt.createQmlObject('import QtQuick; Timer { repeat: false }', root)
        t.interval = ms
        t.triggered.connect(function () { fn(); t.destroy() })
        t.start()
    }
    function rhex(n) { var s = ""; for (var i = 0; i < n * 2; i++) s += "0123456789abcdef"[Math.floor(Math.random() * 16)]; return s }
    function unpack(a, L) { var o = []; for (var i = 0; i < L; i++) o.push((a[i >> 3] >> (7 - (i & 7))) & 1); return o }
    function detrend(y, w) {
        var out = []
        for (var i = 0; i < y.length; i++) {
            var s = 0, n = 0
            for (var j = Math.max(0, i - w); j < Math.min(y.length, i + w); j++) { s += y[j]; n++ }
            out.push(y[i] - s / n)
        }
        return out
    }
    function go(i) { flick.contentY = 0; step = i }

    // ── reusable pieces ────────────────────────────────────────────────
    component Kicker: Text { color: c.dim; font.pixelSize: 11; font.letterSpacing: 2.4; font.capitalization: Font.AllUppercase }
    component Title: Text { color: c.ink; font.pixelSize: 40; font.weight: Font.Light; font.letterSpacing: -0.8; wrapMode: Text.WordWrap; Layout.fillWidth: true }
    component Lead: Text { color: c.dim; font.pixelSize: 17; lineHeight: 1.35; wrapMode: Text.WordWrap; textFormat: Text.StyledText; Layout.fillWidth: true; Layout.maximumWidth: 720 }
    component Note: Text { color: c.dim; font.pixelSize: 13; wrapMode: Text.WordWrap; Layout.fillWidth: true; textFormat: Text.StyledText }
    component Label: Text { color: c.dim; font.pixelSize: 11; font.letterSpacing: 2; font.capitalization: Font.AllUppercase; font.weight: Font.Medium }

    component Btn: Rectangle {
        id: b
        property string label
        property bool active: true
        signal clicked
        implicitWidth: bt.implicitWidth + 40; implicitHeight: 44
        color: bma.containsMouse && active ? "transparent" : c.ink
        border.color: c.ink
        opacity: active ? 1 : 0.3
        Text { id: bt; anchors.centerIn: parent; text: b.label.toUpperCase(); font.pixelSize: 12; font.letterSpacing: 1.5
               color: bma.containsMouse && b.active ? c.ink : "#000" }
        MouseArea { id: bma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: if (b.active) b.clicked() }
    }

    component Verdict: ColumnLayout {
        property string kind: ""
        property string title: ""
        property string sub: ""
        visible: kind !== ""
        spacing: 12
        Layout.fillWidth: true
        Rectangle { Layout.fillWidth: true; height: 1; color: c.line }
        Flow {
            Layout.fillWidth: true; spacing: 14
            Text { text: title; font.pixelSize: 15; color: kind === "ok" ? c.ok : kind === "bad" ? c.bad : c.ink }
            Text { text: sub; font.pixelSize: 15; color: c.dim; visible: sub !== ""; wrapMode: Text.WordWrap; width: Math.min(implicitWidth, parent.width) }
        }
    }

    component Card: Rectangle {
        default property alias content: inner.data
        color: c.bg; border.color: c.line
        Layout.fillWidth: true; Layout.fillHeight: true; Layout.preferredWidth: 1
        implicitHeight: inner.implicitHeight + 56
        ColumnLayout { id: inner; anchors.fill: parent; anchors.margins: 28; spacing: 14 }
    }

    // A recorded sample: drag it onto a DropArea with the same key, or click it.
    component Sample: Item {
        id: smp
        property string kind
        property string title
        property string file
        property string thumb: ""
        property string ico: ""
        property bool used: false
        signal use
        Layout.fillWidth: true
        implicitHeight: 58
        Rectangle {
            id: tile
            property var owner: smp
            width: smp.width; height: smp.height
            anchors.horizontalCenter: smp.horizontalCenter; anchors.verticalCenter: smp.verticalCenter
            color: c.bg; border.color: smp.used || sma.containsMouse ? (smp.used ? c.ink : c.line2) : c.line
            Drag.active: sma.drag.active
            Drag.keys: [smp.kind]
            Drag.hotSpot.x: 30; Drag.hotSpot.y: 29
            RowLayout {
                anchors.fill: parent; anchors.margins: 10; spacing: 14
                Rectangle {
                    Layout.preferredWidth: 48; Layout.preferredHeight: 36; color: "#000"; border.color: smp.thumb ? "transparent" : c.line2
                    Image { anchors.fill: parent; source: smp.thumb; fillMode: Image.PreserveAspectCrop; visible: smp.thumb !== "" }
                    Text { anchors.centerIn: parent; text: smp.ico; color: c.dim; font.pixelSize: 10; font.family: "monospace"; visible: smp.thumb === "" }
                }
                ColumnLayout {
                    spacing: 2; Layout.fillWidth: true
                    Text { text: smp.title; color: c.ink; font.pixelSize: 14 }
                    Text { text: smp.file; color: c.faint; font.pixelSize: 11; font.family: "monospace"; elide: Text.ElideRight; Layout.fillWidth: true }
                }
            }
            MouseArea {
                id: sma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.OpenHandCursor
                drag.target: tile
                onClicked: smp.use()
                onReleased: if (tile.Drag.active) tile.Drag.drop()
            }
            states: State {
                when: sma.drag.active
                ParentChange { target: tile; parent: root }
                AnchorChanges { target: tile; anchors.horizontalCenter: undefined; anchors.verticalCenter: undefined }
                PropertyChanges { target: tile; opacity: 0.85; z: 100 }
            }
        }
    }

    // ── header ─────────────────────────────────────────────────────────
    Rectangle {
        id: header
        anchors { left: parent.left; right: parent.right; top: parent.top }
        height: 60; color: c.bg; z: 5
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: c.line }
        Row {
            anchors.left: parent.left; anchors.leftMargin: 40; height: parent.height; spacing: 28
            Row { anchors.verticalCenter: parent.verticalCenter
                  Text { text: "NEURO"; color: c.ink; font.pixelSize: 13; font.letterSpacing: 4 }
                  Text { text: "LOCK"; color: c.dim; font.pixelSize: 13; font.letterSpacing: 4 } }
            Row {
                height: parent.height
                Repeater {
                    model: ["Start", "Scan", "Pulse", "Key", "Consent", "Attacks"]
                    Item {
                        width: nt.implicitWidth + 24; height: parent.height
                        Text { id: nt; anchors.centerIn: parent; text: modelData.toUpperCase(); font.pixelSize: 11; font.letterSpacing: 1.6
                               color: root.step === index || nma.containsMouse ? c.ink : c.faint }
                        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: c.ink; visible: root.step === index }
                        MouseArea { id: nma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: root.go(index) }
                    }
                }
            }
        }
    }

    // ── pages ──────────────────────────────────────────────────────────
    Flickable {
        id: flick
        anchors { left: parent.left; right: parent.right; top: header.bottom; bottom: parent.bottom }
        contentHeight: page.implicitHeight + 120
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ColumnLayout {
            id: page
            x: Math.max(40, (flick.width - width) / 2); y: 56
            width: Math.min(1120, flick.width - 80)
            spacing: 0

            // 0 START ----------------------------------------------------
            ColumnLayout {
                visible: root.step === 0; spacing: 26; Layout.fillWidth: true; Layout.topMargin: 60
                Kicker { text: "Proof of personhood" }
                Text { text: "Prove you're one live human.\nWithout handing over your body."; color: c.ink; font.pixelSize: 60; font.weight: Font.Light
                       font.letterSpacing: -1.6; lineHeight: 1.05; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                Lead { text: "Asha enrolled at a Field Station two weeks ago. Today someone walks up claiming to be her. Follow the station as it decides. The sensors aren't connected yet, so you'll drag in recorded scans and EEG. Everything the station computes from them is real." }
                Btn { label: "Begin"; onClicked: root.go(1) }
            }

            // 1 SCAN -----------------------------------------------------
            ColumnLayout {
                id: scanPage
                visible: root.step === 1; spacing: 0; Layout.fillWidth: true
                property string img: ""
                property bool pix: false
                property string cap: ""
                property bool ready: false
                Kicker { text: "01 · Scan" }
                Title { text: "Who's at the station?"; Layout.topMargin: 20 }
                Lead { text: "Drag a finger scan into the scanner."; Layout.topMargin: 12; Layout.bottomMargin: 36 }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true
                    Card {
                        Rectangle {
                            id: scanner
                            Layout.fillWidth: true; Layout.preferredHeight: width * 0.75
                            color: "#050505"; border.color: scanDrop.containsDrag ? c.ink : c.line; clip: true
                            Image { id: scanImg; anchors.fill: parent; anchors.margins: 1; fillMode: Image.PreserveAspectFit
                                    source: scanPage.img; smooth: !scanPage.pix; opacity: 0
                                    Behavior on opacity { NumberAnimation { duration: 700 } } }
                            Column {
                                anchors.centerIn: parent; spacing: 10; visible: scanPage.img === "" && !sweep.running
                                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "+"; color: c.dim; font.pixelSize: 36; font.weight: Font.ExtraLight }
                                Text { text: "DROP A SCAN HERE"; color: c.faint; font.pixelSize: 11; font.letterSpacing: 2.2 }
                            }
                            Rectangle { id: sweepLine; width: parent.width; height: 1; color: "#fff"; y: -10; visible: sweep.running }
                            SequentialAnimation {
                                id: sweep; loops: 2
                                NumberAnimation { target: sweepLine; property: "y"; from: 0; to: scanner.height; duration: 800; easing.type: Easing.InOutQuad }
                                NumberAnimation { target: sweepLine; property: "y"; from: scanner.height; to: 0; duration: 800; easing.type: Easing.InOutQuad }
                            }
                            DropArea { id: scanDrop; anchors.fill: parent; keys: ["vein"]; onDropped: (drop) => drop.source.owner.use() }
                        }
                        Note { text: scanPage.cap; Layout.preferredHeight: 20 }
                    }
                    Card {
                        Sample { id: sAsha; kind: "vein"; title: "Asha"; file: "FV-USM 2nd_session/019_3/01.jpg"; thumb: nl.asha.s2.raw
                                 onUse: { sAsha.used = true; sBob.used = false; root.loadScan("asha") } }
                        Sample { id: sBob; kind: "vein"; title: "Someone else (Bob)"; file: "FV-USM 2nd_session/096_3/01.jpg"; thumb: nl.bob.s2.raw
                                 onUse: { sBob.used = true; sAsha.used = false; root.loadScan("bob") } }
                        Note { text: "Real near-infrared scans from the FV-USM database, taken about two weeks after enrolment."; Layout.topMargin: 6 }
                        Verdict { id: scanV }
                        Btn { label: "Next"; active: scanPage.ready; onClicked: root.go(2); Layout.topMargin: 10 }
                        Item { Layout.fillHeight: true }
                    }
                }
            }

            // 2 PULSE ----------------------------------------------------
            ColumnLayout {
                visible: root.step === 2; spacing: 0; Layout.fillWidth: true
                Kicker { text: "02 · Alive" }
                Title { text: "Is the finger alive?"; Layout.topMargin: 20 }
                Lead { text: "Every heartbeat makes the finger slightly darker. A printed photo doesn't do that. Drag in a finger video."; Layout.topMargin: 12; Layout.bottomMargin: 36 }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true
                    Card {
                        Rectangle {
                            Layout.fillWidth: true; Layout.preferredHeight: 200; color: "transparent"
                            border.color: pulseDrop.containsDrag ? c.ink : "transparent"
                            Canvas {
                                id: pulseCv; anchors.fill: parent
                                property var ys: []
                                property color col: c.ink
                                onPaint: {
                                    var g = getContext("2d"); g.clearRect(0, 0, width, height)
                                    if (ys.length < 2) {
                                        g.fillStyle = c.faint; g.font = "11px sans-serif"; g.fillText("DROP A FINGER VIDEO HERE", width / 2 - 80, height / 2); return
                                    }
                                    var mn = Math.min.apply(null, ys), mx = Math.max.apply(null, ys); if (mx - mn < 1e-9) mx = mn + 1
                                    g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath()
                                    for (var i = 0; i < ys.length; i++) {
                                        var X = i / (ys.length - 1) * width, Y = height - 8 - (ys[i] - mn) / (mx - mn) * (height - 16)
                                        if (i) g.lineTo(X, Y); else g.moveTo(X, Y)
                                    }
                                    g.stroke()
                                }
                            }
                            DropArea { id: pulseDrop; anchors.fill: parent; keys: ["pulse"]; onDropped: (drop) => drop.source.owner.use() }
                        }
                        Sample { id: pLive; kind: "pulse"; title: "Live finger"; file: "synthetic clip · 10 s"; ico: "MP4"
                                 onUse: { pLive.used = true; pPhoto.used = false; root.runPulse("live") } }
                        Sample { id: pPhoto; kind: "pulse"; title: "Printed photo"; file: "synthetic clip · 10 s"; ico: "MP4"
                                 onUse: { pPhoto.used = true; pLive.used = false; root.runPulse("photo") } }
                        Note { text: "Synthetic clips (simulated). We have no recorded finger videos yet; the detector is real." }
                    }
                    Card {
                        Label { text: "Heart rate" }
                        Text { id: bpmT; text: "—"; color: c.ink; font.pixelSize: 56; font.weight: Font.Light }
                        Verdict { id: pulseV }
                        Btn { id: next2; label: "Next"; active: false; onClicked: root.go(3); Layout.topMargin: 14 }
                        Item { Layout.fillHeight: true }
                    }
                }
            }

            // 3 UNLOCK ---------------------------------------------------
            ColumnLayout {
                visible: root.step === 3; spacing: 0; Layout.fillWidth: true
                Kicker { text: "03 · Unlock" }
                Title { text: "Do the veins open Asha's lock?"; Layout.topMargin: 20 }
                Lead { Layout.topMargin: 12; Layout.bottomMargin: 36
                       text: "At enrolment, Asha's veins were turned into bits and used to lock a secret. Her vein image was then deleted. A new scan never matches exactly, so some bits flip. The lock tolerates up to <b>" + nl.commit[0].t + "</b> flips in each of the 6 blocks. Same person: few flips. Different person: far too many." }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true
                    Card {
                        Label { id: bitTitle; text: "Bit flips since enrolment" }
                        Canvas {
                            id: bitCv
                            Layout.fillWidth: true; Layout.preferredHeight: width * 186 / 765
                            property var bits: []
                            property var err: []
                            property int upto: 0
                            property int n: 255
                            onPaint: {
                                var g = getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, width, height)
                                if (!bits.length) return
                                var nb = bits.length / n, cw = width / n, rh = height / nb
                                for (var j = 0; j < Math.min(upto, bits.length); j++) {
                                    var b = Math.floor(j / n), i = j % n
                                    g.fillStyle = err[j] ? c.bad : (bits[j] ? "#3a3a3a" : "#161616")
                                    g.fillRect(i * cw, b * rh, Math.ceil(cw), rh - 1)
                                }
                            }
                        }
                        Repeater {
                            id: blkRep
                            model: 6
                            RowLayout {
                                property int errs: -1
                                Layout.fillWidth: true; spacing: 12; Layout.topMargin: index === 0 ? 14 : 0
                                Text { text: "block " + (index + 1); color: c.ink; font.pixelSize: 12; font.family: "monospace"; Layout.preferredWidth: 64 }
                                Item {
                                    Layout.fillWidth: true; height: 6
                                    Rectangle { anchors.fill: parent; color: "#1a1a1a" }
                                    Rectangle { height: parent.height; color: errs <= nl.commit[0].t ? c.ok : c.bad
                                                width: errs < 0 ? 0 : Math.min(1, errs / 90) * parent.width
                                                Behavior on width { NumberAnimation { duration: 600; easing.type: Easing.OutCubic } } }
                                    Rectangle { x: nl.commit[0].t / 90 * parent.width; y: -5; width: 1; height: 16; color: c.dim }
                                }
                                Text { text: errs < 0 ? "" : errs; color: c.ink; font.pixelSize: 12; font.family: "monospace"; Layout.preferredWidth: 30 }
                            }
                        }
                        Note { text: "Vertical line = most flips a block can have and still open."; Layout.topMargin: 4 }
                    }
                    Card {
                        Text { id: lockLog; color: c.dim; font.pixelSize: 13; font.family: "monospace"; lineHeight: 1.6; textFormat: Text.StyledText
                               wrapMode: Text.WordWrap; Layout.fillWidth: true; Layout.minimumHeight: 120 }
                        Verdict { id: lockV }
                        RowLayout {
                            spacing: 10; Layout.topMargin: 10
                            Btn { id: lockBtn; label: "Try to unlock"; onClicked: root.lockRun() }
                            Btn { id: next3; label: "Next"; visible: false; onClicked: root.go(4) }
                        }
                        Item { Layout.fillHeight: true }
                    }
                }
            }

            // 4 CONSENT --------------------------------------------------
            ColumnLayout {
                id: conPage
                visible: root.step === 4; spacing: 0; Layout.fillWidth: true
                property bool eeg: false
                property bool calm: false
                Kicker { text: "04 · Consent" }
                Title { text: "Does she mean it?"; Layout.topMargin: 20 }
                Lead { Layout.topMargin: 12; Layout.bottomMargin: 24
                       text: "She is asked to approve an action. Both buttons flicker, each at a new random rate every 1.5 seconds. Whichever one she looks at, her brain echoes its rhythm. That shows a live person choosing, right now, for this action. Drag in her EEG, press start, and hold your mouse on a button." }
                RowLayout {
                    spacing: 10; Layout.bottomMargin: 24
                    Rectangle { width: 1; height: 22; color: c.ink }
                    Text { text: "Flickers at 6–15 Hz."; color: c.dim; font.pixelSize: 13 }
                    Text { text: conPage.calm ? "[x] Reduce flicker" : "[ ] Reduce flicker"; color: c.dim; font.pixelSize: 13
                           MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: conPage.calm = !conPage.calm } }
                }
                Rectangle {
                    Layout.fillWidth: true; implicitHeight: 64; color: c.bg; border.color: c.line
                    Text { anchors.verticalCenter: parent.verticalCenter; x: 24; text: "transfer 50 FS tokens to field-station-vote"; color: c.ink; font.pixelSize: 17; font.family: "monospace" }
                }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true; Layout.topMargin: 28
                    Repeater {
                        model: [["CONFIRM", 0], ["DENY", 1]]
                        Rectangle {
                            id: fb
                            property bool lit: con.running && (modelData[1] === 0 ? con.onC : con.onD)
                            Layout.fillWidth: true; Layout.preferredWidth: 1; implicitHeight: 170
                            color: lit ? (conPage.calm ? "#2a2a2a" : c.ink) : c.bg; border.color: c.line
                            Rectangle { anchors.fill: parent; anchors.margins: 10; color: "transparent"; border.color: c.dim; visible: fma.containsMouse }
                            Column {
                                anchors.centerIn: parent; spacing: 8
                                Text { anchors.horizontalCenter: parent.horizontalCenter; text: modelData[0]; font.pixelSize: 22; font.letterSpacing: 6
                                       color: fb.lit && !conPage.calm ? "#000" : c.ink }
                                Text { anchors.horizontalCenter: parent.horizontalCenter; font.pixelSize: 11; color: c.dim
                                       text: con.running ? pm.freqs[modelData[1] === 0 ? con.cI : con.dI].toFixed(1) + " Hz" : "" }
                            }
                            MouseArea { id: fma; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.CrossCursor
                                        onContainsMouseChanged: con.gaze = containsMouse ? modelData[1] : (con.gaze === modelData[1] ? -1 : con.gaze) }
                        }
                    }
                }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true; Layout.topMargin: 12
                    Repeater {
                        id: segRep; model: 4
                        Rectangle {
                            property string res: ""
                            Layout.fillWidth: true; Layout.preferredWidth: 1; implicitHeight: 38; color: c.bg; border.color: c.line
                            Text { anchors.centerIn: parent; font.pixelSize: 11
                                   text: res === "c" ? "Confirm" : res === "d" ? "Deny" : res === "-" ? "Unclear" : (con.running && con.seg === index ? "…" : (index + 1))
                                   color: res === "c" ? c.ok : res === "d" ? c.bad : res === "-" ? c.ink : c.faint }
                            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: c.ink; visible: con.running && con.seg === index }
                        }
                    }
                }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true; Layout.topMargin: 28
                    Card {
                        Rectangle {
                            Layout.fillWidth: true; implicitHeight: 54; color: "transparent"
                            border.color: eegDrop.containsDrag || conPage.eeg ? c.ink : c.line2
                            Text { anchors.centerIn: parent; text: conPage.eeg ? "Asha · EEG loaded" : "DROP EEG HERE"
                                   color: conPage.eeg ? c.ink : c.faint; font.pixelSize: conPage.eeg ? 13 : 11; font.letterSpacing: conPage.eeg ? 0 : 2 }
                            DropArea { id: eegDrop; anchors.fill: parent; keys: ["eeg"]; onDropped: (drop) => drop.source.owner.use() }
                        }
                        Sample { id: eegS; kind: "eeg"; title: "Asha · resting EEG"; file: "PhysioNet subject " + nl.asha.id[1] + " · 8 channels"; ico: "EEG"
                                 onUse: { eegS.used = true; conPage.eeg = true } }
                        Note { text: "Real resting EEG (PhysioNet) with a simulated flicker response added. The detector is real." }
                        Btn { label: "Start"; Layout.topMargin: 6; active: !con.running; onClicked: root.startConsent(1) }
                        Item { Layout.fillHeight: true }
                    }
                    Card {
                        Label { text: "Brain response at each rate" }
                        Canvas {
                            id: rhoCv
                            Layout.fillWidth: true; Layout.preferredHeight: 170
                            property var rhos: []
                            property int cI: -1
                            property int dI: -1
                            onPaint: {
                                var g = getContext("2d"); g.clearRect(0, 0, width, height)
                                var F = pm.freqs, bw = width / F.length, H = height
                                function yv(r) { return H - 22 - r * (H - 34) }
                                g.font = "11px sans-serif"
                                for (var i = 0; i < F.length; i++) {
                                    var r = rhos.length ? rhos[i] : 0
                                    g.fillStyle = i === cI ? c.ok : i === dI ? c.bad : "#2a2a2a"
                                    g.fillRect(i * bw + 8, yv(r), bw - 16, H - 22 - yv(r))
                                    g.fillStyle = "#666"; g.fillText(F[i].toFixed(1) + " Hz", i * bw + 10, H - 6)
                                }
                                g.strokeStyle = "#666"; g.setLineDash([4, 4]); g.beginPath()
                                g.moveTo(0, yv(pm.rho_thr)); g.lineTo(width, yv(pm.rho_thr)); g.stroke(); g.setLineDash([])
                            }
                        }
                        Verdict { id: conV }
                        RowLayout {
                            id: chainRow; spacing: 28; visible: false; Layout.topMargin: 8
                            property var blocks: []
                            Repeater {
                                model: chainRow.blocks
                                Rectangle {
                                    implicitWidth: 220; implicitHeight: 86; color: c.bg; border.color: c.line
                                    Column {
                                        anchors.fill: parent; anchors.margins: 14; spacing: 4
                                        Text { text: modelData[0]; color: c.ink; font.pixelSize: 11; font.letterSpacing: 2 }
                                        Text { text: modelData[1]; color: c.faint; font.pixelSize: 11 }
                                        Text { text: modelData[2].slice(0, 22) + "…"; color: c.ink; font.pixelSize: 11; font.family: "monospace" }
                                    }
                                }
                            }
                        }
                        Item { Layout.fillHeight: true }
                    }
                }
                Btn { label: "Next"; onClicked: root.go(5); Layout.topMargin: 24 }
            }

            // 5 ATTACKS --------------------------------------------------
            ColumnLayout {
                visible: root.step === 5; spacing: 0; Layout.fillWidth: true
                Kicker { text: "05 · Attacks" }
                Title { text: "Try to fool it"; Layout.topMargin: 20 }
                Lead { text: "Each attack passes through every check. One of them stops it."; Layout.topMargin: 12; Layout.bottomMargin: 28 }
                Btn { label: "Run attacks"; onClicked: root.runAttacks() }
                ColumnLayout {
                    spacing: 0; Layout.fillWidth: true; Layout.topMargin: 28
                    Repeater {
                        id: attRep
                        model: root.attacks
                        ColumnLayout {
                            property bool shown: false
                            spacing: 0; Layout.fillWidth: true
                            Rectangle { Layout.fillWidth: true; height: 1; color: c.line }
                            RowLayout {
                                Layout.fillWidth: true; Layout.topMargin: 18; Layout.bottomMargin: 18
                                Text { text: modelData[0]; color: c.ink; font.pixelSize: 16; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                                Column {
                                    visible: shown; spacing: 2
                                    Text { anchors.right: parent.right; text: "Stopped · " + modelData[1]; color: c.bad; font.pixelSize: 13 }
                                    Text { anchors.right: parent.right; text: modelData[2]; color: c.dim; font.pixelSize: 12 }
                                }
                            }
                        }
                    }
                    Rectangle { Layout.fillWidth: true; height: 1; color: c.line }
                }
                RowLayout {
                    spacing: -1; Layout.fillWidth: true; Layout.topMargin: 40
                    Card { Label { text: "Wrong person accepted" } Text { text: "1 in 2970"; color: c.ink; font.pixelSize: 44; font.weight: Font.Light } }
                    Card { Label { text: "Duplicate sign-ups caught" } Text { text: "98%"; color: c.ink; font.pixelSize: 44; font.weight: Font.Light } }
                    Card { Label { text: "Vein error rate" } Text { text: "0.4%"; color: c.ink; font.pixelSize: 44; font.weight: Font.Light } }
                }
                Note { Layout.topMargin: 20; Layout.maximumWidth: 820
                       text: "Measured on held-out people. Still simulated: brain flicker responses and finger pulse. Not built yet: the scanner hardware. An attacker who controls the capture software could fake the pulse and the brain signal; stopping that needs trusted sensors." }
            }
        }
    }

    // ── logic: scan ────────────────────────────────────────────────────
    property int scanRun: 0
    function loadScan(w) {
        who = w; resetDownstream()
        var run = ++scanRun, s = (w === "asha" ? nl.asha.s2 : nl.bob.s2)
        scanPage.ready = false; scanV.kind = ""; scanImg.opacity = 0; scanPage.img = ""
        scanPage.cap = "Scanning…"; sweep.restart()
        var steps = [[s.raw, false, "Near-infrared image: veins absorb the light"], [s.roi, true, "Cropped and straightened"],
                     [s.enh, true, "Contrast enhanced"], [s.maxcurv, true, "Vein pattern extracted"]]
        steps.forEach(function (st, k) {
            after(3200 + k * 1500, function () {
                if (run !== scanRun) return
                scanImg.opacity = 0; scanPage.img = st[0]; scanPage.pix = st[1]; scanImg.opacity = 1; scanPage.cap = st[2]
            })
        })
        after(3200 + 4 * 1500 - 600, function () {
            if (run !== scanRun) return
            scanV.kind = "ok"; scanV.title = "Vein pattern ready"; scanV.sub = ""; scanPage.ready = true
        })
    }
    function resetDownstream() {
        lockLog.text = ""; lockV.kind = ""; lockBtn.visible = true; lockBtn.label = "Try to unlock"; next3.visible = false
        bitCv.bits = []; bitCv.requestPaint()
        for (var b = 0; b < 6; b++) blkRep.itemAt(b).errs = -1
        conV.kind = ""; chainRow.visible = false
    }

    // ── logic: pulse ───────────────────────────────────────────────────
    Timer {
        id: pulseT; interval: 35; repeat: true
        property var tr: []
        property int i: 30
        property var p
        onTriggered: {
            i += 6
            pulseCv.ys = root.detrend(tr.slice(Math.max(0, i - 180), i), 15).slice(15, -15); pulseCv.requestPaint()
            if (i >= tr.length) {
                stop()
                if (p.ok) { bpmT.text = p.bpm.toFixed(0); pulseV.kind = "ok"; pulseV.title = "Heartbeat found"; pulseV.sub = ""; next2.active = true }
                else { pulseV.kind = "bad"; pulseV.title = "No heartbeat"; pulseV.sub = "a photo has no blood flow" }
            }
        }
    }
    function runPulse(kind) {
        var p = nl.pulse[kind]
        pulseV.kind = ""; bpmT.text = "—"; pulseCv.col = kind === "live" ? c.ink : "#666666"
        pulseT.p = p; pulseT.tr = p.trace; pulseT.i = 30; pulseT.restart()
    }

    // ── logic: unlock ──────────────────────────────────────────────────
    Timer {
        id: bitT; interval: 16; repeat: true
        property var done
        onTriggered: { bitCv.upto += 60; bitCv.requestPaint(); if (bitCv.upto >= bitCv.bits.length) { stop(); done() } }
    }
    function lockFinger(C, P, cb) {
        bitTitle.text = "Finger " + C.finger + ": bit flips since enrolment"
        for (var b = 0; b < 6; b++) blkRep.itemAt(b).errs = -1
        bitCv.n = C.n; bitCv.bits = unpack(C.wbits, C.L); bitCv.err = unpack(P.errbits, C.L); bitCv.upto = 0
        bitT.done = function () {
            P.blocks.forEach(function (e, b) { after(b * 160, function () { blkRep.itemAt(b).errs = e }) })
            after(6 * 160 + 500, cb)
        }
        bitT.restart()
    }
    function lockRun() {
        if (!who) { lockV.kind = "warn"; lockV.title = "Scan a finger first"; lockV.sub = ""; return }
        lockBtn.visible = false; lockLog.text = ""; lockV.kind = ""
        var fingers = nl.commit
        function attempt(k) {
            var C = fingers[k], P = who === "asha" ? C.asha : C.bob
            lockFinger(C, P, function () {
                lockLog.text += "finger " + C.finger + ": " + (P.ber * 100).toFixed(0) + "% of bits flipped → " + (P.ok ? "opens" : "stays locked") + "<br>"
                if (P.ok) {
                    lockLog.text += "secret recovered → identity key " + rhex(6) + "…<br>"
                    lockV.kind = "ok"; lockV.title = "It's Asha"; lockV.sub = "Her veins opened the lock. No vein image was stored."
                    next3.visible = true; return
                }
                if (k + 1 < fingers.length) { after(1000, function () { attempt(k + 1) }); return }
                lockV.kind = "bad"; lockV.title = "Rejected"; lockV.sub = "These veins don't open Asha's lock. The station stops here."
                lockBtn.visible = true; lockBtn.label = "Try again"
            })
        }
        attempt(0)
    }

    // ── logic: consent (also proves liveness) ──────────────────────────
    Timer {
        id: con; interval: 16; repeat: true
        property int seg: 0
        property int cI: 0
        property int dI: 1
        property int gaze: -1
        property real t0: 0
        property var cnt: [0, 0, 0]
        property var ans: []
        property int attempt: 1
        property int segN: 0
        property bool onC: false
        property bool onD: false
        onTriggered: {
            var now = Date.now(), t = now / 1000
            onC = (t * pm.freqs[cI]) % 1 < 0.5; onD = (t * pm.freqs[dI]) % 1 < 0.5
            var k = gaze === 0 ? 0 : gaze === 1 ? 1 : 2; cnt[k]++
            if (now - t0 < pm.seg_s * 1000) return
            var tot = cnt[0] + cnt[1] + cnt[2]
            var ti = cnt[0] / tot > 0.6 ? cI : cnt[1] / tot > 0.6 ? dI : pm.freqs.length
            var r = nl.ssvep.grid[(segN++) % nl.ssvep.grid.length][ti].rhos, best = 0
            for (var j = 1; j < r.length; j++) if (r[j] > r[best]) best = j
            var a = r[best] >= pm.rho_thr && best === cI ? "c" : r[best] >= pm.rho_thr && best === dI ? "d" : "-"
            ans.push(a); segRep.itemAt(seg).res = a
            rhoCv.rhos = r; rhoCv.cI = cI; rhoCv.dI = dI; rhoCv.requestPaint()
            if (seg + 1 < pm.K) { root.nextSegment(seg + 1); return }
            stop(); root.finishConsent()
        }
    }
    function pick(excl) { var a; do { a = Math.floor(Math.random() * pm.freqs.length) } while (a === excl); return a }
    function nextSegment(i) { con.seg = i; con.cI = pick(-1); con.dI = pick(con.cI); con.cnt = [0, 0, 0]; con.t0 = Date.now() }
    function startConsent(attempt) {
        if (!conPage.eeg) { conV.kind = "warn"; conV.title = "Drop the EEG in first"; conV.sub = ""; return }
        conV.kind = ""; chainRow.visible = false; con.attempt = attempt; con.ans = []
        for (var i = 0; i < 4; i++) segRep.itemAt(i).res = ""
        nextSegment(0); con.start()
    }
    function finishConsent() {
        var ans = con.ans
        function all(x) { return ans.every(function (a) { return a === x }) }
        if (all("c")) {
            conV.kind = "ok"; conV.title = "Confirmed"; conV.sub = "signed and recorded"
            var r1 = Sha.hash("register" + rhex(32)), r2 = Sha.hash(r1 + "consent" + rhex(32))
            chainRow.blocks = [["ENROLLED", "two weeks ago", r1], ["CONSENT", "just now", r2]]; chainRow.visible = true
        } else if (all("d")) { conV.kind = "bad"; conV.title = "Denied"; conV.sub = "nothing is signed" }
        else if (con.attempt < 3) {
            conV.kind = "warn"; conV.title = "Unclear, trying again (" + (con.attempt + 1) + " of 3)"; conV.sub = ""
            after(1400, function () { startConsent(con.attempt + 1) })
        } else { conV.kind = "warn"; conV.title = "No clear answer"; conV.sub = "nothing is signed" }
    }

    // ── logic: attacks ─────────────────────────────────────────────────
    readonly property var attacks: [
        ["Someone else uses their own finger", "Unlock", (nl.commit[0].bob.ber * 100).toFixed(0) + "% of bits flipped: far past the limit"],
        ["Printed photo of Asha's finger", "Pulse", "no heartbeat"],
        ["Asha tries to sign up a second time", "Duplicate check", "matches her existing record (score " + nl.dedup.asha2[0].toFixed(1) + " vs limit " + pm.thr_dedup.toFixed(1) + ")"],
        ["Replay a recording of Asha's earlier \"confirm\"", "Consent", "the old recording follows old flicker rates"],
        ["Pressure Asha: she looks at DENY", "Consent", "nothing is signed"],
        ["Edit the ledger afterwards", "Ledger", "the hash chain breaks"]]
    function runAttacks() {
        for (var i = 0; i < attacks.length; i++) attRep.itemAt(i).shown = false
        for (i = 0; i < attacks.length; i++) (function (k) { after(450 * (k + 1), function () { attRep.itemAt(k).shown = true }) })(i)
    }
}
