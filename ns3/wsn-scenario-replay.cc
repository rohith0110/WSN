/*
 * WSN routing lab: replay one exported field in ns-3 over IEEE 802.15.4 radios.
 *
 * Reads a scenario exported by `python -m wsn.cli ns3-export` (sensor positions, sink and the
 * shortest-path next hop of every sensor). Every round, each sensor sends one reading that is
 * relayed hop by hop to the sink. MAC acknowledgements and retries are off (one attempt per
 * hop, as in the Python simulator). Two traffic modes:
 *   isolated:   each reading travels alone, so only the radio channel can drop it;
 *   contention: all readings of a round start within one second, so CSMA/CA, collisions
 *               and busy channels also drop readings (the Python simulator ignores these).
 *
 * Usage: wsn-scenario-replay --scenario=field.txt --output=replay.csv --mode=isolated
 */

#include "ns3/command-line.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/core-module.h"
#include "ns3/lr-wpan-mac.h"
#include "ns3/lr-wpan-net-device.h"
#include "ns3/lr-wpan-spectrum-value-helper.h"
#include "ns3/mac16-address.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/simulator.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using namespace ns3::lrwpan;

namespace
{
constexpr uint32_t PAYLOAD_BYTES = 100;
constexpr uint16_t PAN_ID = 7;

struct Field
{
    uint32_t sensors{0};
    std::vector<Vector> positions; // sensors first, sink last
    std::vector<int32_t> nextHop;  // per sensor; -1 when no route
};

struct Reading
{
    uint16_t source;
    uint32_t round;
    uint8_t hops;
};

Field
ReadField(const std::string& path)
{
    std::ifstream in(path);
    NS_ABORT_MSG_UNLESS(in.is_open(), "Cannot open scenario file " << path);
    Field field;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream row(line);
        std::string kind;
        row >> kind;
        if (kind == "sensors")
        {
            row >> field.sensors;
            field.positions.assign(field.sensors + 1, Vector());
            field.nextHop.assign(field.sensors, -1);
        }
        else if (kind == "position")
        {
            uint32_t id;
            double x;
            double y;
            row >> id >> x >> y;
            field.positions.at(id) = Vector(x, y, 0);
        }
        else if (kind == "next")
        {
            uint32_t id;
            int32_t hop;
            row >> id >> hop;
            field.nextHop.at(id) = hop;
        }
    }
    NS_ABORT_MSG_IF(field.sensors == 0, "Scenario file has no sensors");
    return field;
}

class Replay
{
  public:
    Replay(const Field& field, double exponent, double fadingM)
        : m_field(field),
          m_delivered(0),
          m_hopsDelivered(0)
    {
        Ptr<MultiModelSpectrumChannel> channel = CreateObject<MultiModelSpectrumChannel>();
        Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
        loss->SetAttribute("Exponent", DoubleValue(exponent));
        if (fadingM > 0)
        {
            Ptr<NakagamiPropagationLossModel> fading = CreateObject<NakagamiPropagationLossModel>();
            fading->SetAttribute("m0", DoubleValue(fadingM));
            fading->SetAttribute("m1", DoubleValue(fadingM));
            fading->SetAttribute("m2", DoubleValue(fadingM));
            loss->SetNext(fading);
        }
        channel->AddPropagationLossModel(loss);
        LrWpanSpectrumValueHelper helper;
        for (uint32_t id = 0; id <= field.sensors; id++)
        {
            Ptr<Node> node = CreateObject<Node>();
            Ptr<LrWpanNetDevice> device = CreateObject<LrWpanNetDevice>();
            device->SetAddress(Mac16Address(static_cast<uint16_t>(id + 1)));
            device->SetChannel(channel);
            node->AddDevice(device);
            device->GetMac()->SetPanId(PAN_ID);
            Ptr<ConstantPositionMobilityModel> mobility =
                CreateObject<ConstantPositionMobilityModel>();
            mobility->SetPosition(field.positions[id]);
            device->GetPhy()->SetMobility(mobility);
            device->GetPhy()->SetTxPowerSpectralDensity(helper.CreateTxPowerSpectralDensity(0, 11));
            device->GetMac()->SetMcpsDataIndicationCallback(
                MakeCallback(&Replay::Received, this).Bind(id));
            m_devices.push_back(device);
        }
    }

    void Send(uint32_t from, const Reading& reading)
    {
        if (from >= m_field.sensors || m_field.nextHop[from] < 0)
        {
            return; // no route: dropped
        }
        uint8_t buffer[PAYLOAD_BYTES] = {};
        std::memcpy(buffer, &reading, sizeof(reading));
        McpsDataRequestParams params;
        params.m_srcAddrMode = SHORT_ADDR;
        params.m_dstAddrMode = SHORT_ADDR;
        params.m_dstPanId = PAN_ID;
        params.m_dstAddr = Mac16Address(static_cast<uint16_t>(m_field.nextHop[from] + 1));
        params.m_msduHandle = 0;
        params.m_txOptions = 0; // no acknowledgement, no retry
        m_devices[from]->GetMac()->McpsDataRequest(params, Create<Packet>(buffer, PAYLOAD_BYTES));
    }

    uint32_t Delivered() const
    {
        return m_delivered;
    }

    uint64_t HopsDelivered() const
    {
        return m_hopsDelivered;
    }

  private:
    void Received(uint32_t at, McpsDataIndicationParams, Ptr<Packet> packet)
    {
        Reading reading;
        packet->CopyData(reinterpret_cast<uint8_t*>(&reading), sizeof(reading));
        reading.hops++;
        if (at == m_field.sensors)
        {
            m_delivered++;
            m_hopsDelivered += reading.hops;
            return;
        }
        if (reading.hops > 2 * m_field.sensors)
        {
            return;
        }
        Simulator::ScheduleNow(&Replay::Send, this, at, reading);
    }

    Field m_field;
    std::vector<Ptr<LrWpanNetDevice>> m_devices;
    uint32_t m_delivered;
    uint64_t m_hopsDelivered;
};
} // namespace

int
main(int argc, char* argv[])
{
    std::string scenario;
    std::string output = "wsn-scenario-replay.csv";
    std::string mode = "isolated";
    double exponent = 3.0;
    double fadingM = 1.0;
    uint32_t rounds = 50;
    uint32_t seed = 2000;

    CommandLine cmd(__FILE__);
    cmd.AddValue("scenario", "scenario file from `wsn.cli ns3-export`", scenario);
    cmd.AddValue("output", "CSV file to write", output);
    cmd.AddValue("mode", "isolated or contention", mode);
    cmd.AddValue("exponent", "log-distance path loss exponent", exponent);
    cmd.AddValue("fadingM", "Nakagami m (1 = Rayleigh); 0 disables fading", fadingM);
    cmd.AddValue("rounds", "rounds of traffic (one reading per sensor per round)", rounds);
    cmd.AddValue("seed", "random seed", seed);
    cmd.Parse(argc, argv);
    NS_ABORT_MSG_IF(scenario.empty(), "--scenario is required");
    NS_ABORT_MSG_UNLESS(mode == "isolated" || mode == "contention",
                        "--mode must be isolated or contention");

    RngSeedManager::SetSeed(seed);
    Field field = ReadField(scenario);
    Replay replay(field, exponent, fadingM);
    Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable>();

    const bool isolated = mode == "isolated";
    const double roundSeconds = isolated ? 0.2 * field.sensors : 1.0;
    for (uint32_t round = 0; round < rounds; round++)
    {
        for (uint32_t source = 0; source < field.sensors; source++)
        {
            double offset = isolated ? 0.2 * source : jitter->GetValue(0.0, 0.9);
            Reading reading{static_cast<uint16_t>(source), round, 0};
            Simulator::Schedule(Seconds(round * roundSeconds + offset),
                                &Replay::Send,
                                &replay,
                                source,
                                reading);
        }
    }
    Simulator::Stop(Seconds(rounds * roundSeconds + 5));
    Simulator::Run();

    uint32_t generated = rounds * field.sensors;
    std::ofstream csv(output);
    csv << "mode,exponent,nakagami_m,rounds,sensors,generated,delivered,pdr,mean_hops\n";
    csv << mode << ',' << exponent << ',' << fadingM << ',' << rounds << ',' << field.sensors
        << ',' << generated << ',' << replay.Delivered() << ','
        << static_cast<double>(replay.Delivered()) / generated << ','
        << (replay.Delivered() ? static_cast<double>(replay.HopsDelivered()) / replay.Delivered()
                               : 0.0)
        << '\n';
    Simulator::Destroy();
    std::cout << mode << ": delivered " << replay.Delivered() << " of " << generated << "\n";
    return 0;
}
