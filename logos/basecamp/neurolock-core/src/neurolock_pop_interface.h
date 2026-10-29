#ifndef NEUROLOCK_POP_INTERFACE_H
#define NEUROLOCK_POP_INTERFACE_H

#include <QObject>
#include <QString>
#include "interface.h"

// Methods the QML tab calls with logos.callModule("neurolock_pop", ...).
// Basecamp marshals every QML argument as QString, so all params are QString.
class NeurolockPopInterface : public PluginInterface
{
public:
    virtual ~NeurolockPopInterface() = default;
    // JSON of the service's /state: enrolled? delegations, jobs, per-layer log
    Q_INVOKABLE virtual QString state() = 0;
    // Start an action: enrol | verify | consent | delegate | revoke | impostor | replay
    // `app` is used by delegate/revoke. Returns {"job": n} or {"error": "..."}.
    Q_INVOKABLE virtual QString start(const QString& action, const QString& app) = 0;
};

#define NeurolockPopInterface_iid "org.neurolock.PopInterface"
Q_DECLARE_INTERFACE(NeurolockPopInterface, NeurolockPopInterface_iid)

#endif
