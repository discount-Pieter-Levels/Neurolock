#include "neurolock_pop_plugin.h"
#include "logos_api.h"

#include <QEventLoop>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <QUrl>
#include <QUrlQuery>

// The QML tab is sandboxed (no network). This core module runs outside the
// sandbox and only ever talks to 127.0.0.1, where logos/pop_service.py runs
// the Neurolock pipeline. Port override: NEUROLOCK_POP_PORT.
NeurolockPopPlugin::NeurolockPopPlugin(QObject* parent)
    : QObject(parent)
{
    const int port = qEnvironmentVariableIsSet("NEUROLOCK_POP_PORT")
        ? qEnvironmentVariableIntValue("NEUROLOCK_POP_PORT") : 47200;
    m_base = QString("http://127.0.0.1:%1").arg(port);
}

NeurolockPopPlugin::~NeurolockPopPlugin() = default;

void NeurolockPopPlugin::initLogos(LogosAPI* api)
{
    logosAPI = api;
    m_net = new QNetworkAccessManager(this);
}

QString NeurolockPopPlugin::request(const QByteArray& verb, const QString& path)
{
    if (!m_net) m_net = new QNetworkAccessManager(this);
    QNetworkRequest req(QUrl(m_base + path));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    QNetworkReply* reply = m_net->sendCustomRequest(req, verb, QByteArray());
    QEventLoop loop;
    QTimer timer;
    timer.setSingleShot(true);
    QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    timer.start(5000);
    loop.exec();
    QString out;
    if (!reply->isFinished()) {
        reply->abort();
        out = R"({"error":"Neurolock service did not answer (is pop_service.py running?)"})";
    } else if (reply->error() != QNetworkReply::NoError && reply->bytesAvailable() == 0) {
        out = QJsonDocument(QJsonObject{{"error", "Neurolock service unreachable: " + reply->errorString()}})
                  .toJson(QJsonDocument::Compact);
    } else {
        out = QString::fromUtf8(reply->readAll());
    }
    reply->deleteLater();
    return out;
}

QString NeurolockPopPlugin::state()
{
    return request("GET", "/state");
}

QString NeurolockPopPlugin::start(const QString& action, const QString& app)
{
    static const QStringList allowed = {"enrol", "verify", "consent", "delegate", "revoke", "impostor", "replay"};
    if (!allowed.contains(action))
        return R"({"error":"unknown action"})";
    QString path = "/" + action;
    if (!app.isEmpty()) {
        QUrlQuery q; q.addQueryItem("app", app);
        path += "?" + q.toString(QUrl::FullyEncoded);
    }
    return request("POST", path);
}
