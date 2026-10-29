#ifndef NEUROLOCK_POP_PLUGIN_H
#define NEUROLOCK_POP_PLUGIN_H

#include <QObject>
#include <QString>
#include "neurolock_pop_interface.h"
#include "logos_api.h"
#include "logos_sdk.h"

class QNetworkAccessManager;

class NeurolockPopPlugin : public QObject, public NeurolockPopInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID NeurolockPopInterface_iid FILE "metadata.json")
    Q_INTERFACES(NeurolockPopInterface PluginInterface)

public:
    explicit NeurolockPopPlugin(QObject* parent = nullptr);
    ~NeurolockPopPlugin() override;

    QString name() const override { return "neurolock_pop"; }
    QString version() const override { return "0.1.0"; }

    Q_INVOKABLE void initLogos(LogosAPI* api);
    Q_INVOKABLE QString state() override;
    Q_INVOKABLE QString start(const QString& action, const QString& app) override;

signals:
    void eventResponse(const QString& eventName, const QVariantList& args);

private:
    // One short HTTP round trip to the local service (it answers immediately;
    // long work runs as a background job there).
    QString request(const QByteArray& verb, const QString& path);
    QNetworkAccessManager* m_net = nullptr;
    QString m_base;
};

#endif
