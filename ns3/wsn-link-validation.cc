/*
 * WSN routing lab: link-level validation against the ns-3 IEEE 802.15.4 (LR-WPAN) model.
 *
 * The Python simulator assumes an abstract per-link success probability that falls with
 * distance. This program measures the packet success rate (PSR) of real 802.15.4 frames
 * (2.4 GHz O-QPSK error model) between two static nodes as a function of distance, under a
 * log-distance path loss with Nakagami fading, and writes one CSV row per configuration.
 *
 * Adapted from src/lr-wpan/examples/lr-wpan-error-distance-plot.cc (GPL-2.0-only).
 *
 * Usage (from the ns-3 root, after copying this file to scratch/):
 *   ./ns3 run "wsn-link-validation --output=psr.csv --exponents=3.0,3.5 --fadingM=1"
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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using namespace ns3::lrwpan;

namespace
{
uint32_t g_received = 0;

void
OnReceive(McpsDataIndicationParams, Ptr<Packet>)
{
    g_received++;
}

std::vector<double>
ParseList(const std::string& text)
{
    std::vector<double> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (!item.empty())
        {
            values.push_back(std::stod(item));
        }
    }
    return values;
}

struct Measurement
{
    uint32_t sent;
    uint32_t received;
};

Measurement
MeasureLink(double distance,
            double exponent,
            double fadingM,
            uint32_t packets,
            uint32_t msduBytes,
            double txPowerDbm,
            double rxSensitivityDbm)
{
    Ptr<Node> sender = CreateObject<Node>();
    Ptr<Node> receiver = CreateObject<Node>();
    Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice>();
    Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice>();
    dev0->SetAddress(Mac16Address("00:01"));
    dev1->SetAddress(Mac16Address("00:02"));

    Ptr<MultiModelSpectrumChannel> channel = CreateObject<MultiModelSpectrumChannel>();
    Ptr<LogDistancePropagationLossModel> pathLoss =
        CreateObject<LogDistancePropagationLossModel>();
    pathLoss->SetAttribute("Exponent", DoubleValue(exponent));
    if (fadingM > 0)
    {
        Ptr<NakagamiPropagationLossModel> fading = CreateObject<NakagamiPropagationLossModel>();
        fading->SetAttribute("m0", DoubleValue(fadingM));
        fading->SetAttribute("m1", DoubleValue(fadingM));
        fading->SetAttribute("m2", DoubleValue(fadingM));
        pathLoss->SetNext(fading);
    }
    channel->AddPropagationLossModel(pathLoss);

    dev0->SetChannel(channel);
    dev1->SetChannel(channel);
    sender->AddDevice(dev0);
    receiver->AddDevice(dev1);

    Ptr<ConstantPositionMobilityModel> mob0 = CreateObject<ConstantPositionMobilityModel>();
    Ptr<ConstantPositionMobilityModel> mob1 = CreateObject<ConstantPositionMobilityModel>();
    mob0->SetPosition(Vector(0, 0, 0));
    mob1->SetPosition(Vector(distance, 0, 0));
    dev0->GetPhy()->SetMobility(mob0);
    dev1->GetPhy()->SetMobility(mob1);

    LrWpanSpectrumValueHelper helper;
    dev0->GetPhy()->SetTxPowerSpectralDensity(helper.CreateTxPowerSpectralDensity(txPowerDbm, 11));
    dev1->GetPhy()->SetRxSensitivity(rxSensitivityDbm);
    dev1->GetMac()->SetMcpsDataIndicationCallback(MakeCallback(&OnReceive));

    McpsDataRequestParams params;
    params.m_srcAddrMode = SHORT_ADDR;
    params.m_dstAddrMode = SHORT_ADDR;
    params.m_dstPanId = 0;
    params.m_dstAddr = Mac16Address("00:02");
    params.m_msduHandle = 0;
    params.m_txOptions = 0; // no acknowledgement and no retries: one attempt per frame

    g_received = 0;
    for (uint32_t i = 0; i < packets; i++)
    {
        Simulator::Schedule(MilliSeconds(20 * (i + 1)),
                            &LrWpanMac::McpsDataRequest,
                            dev0->GetMac(),
                            params,
                            Create<Packet>(msduBytes));
    }
    Simulator::Run();
    Simulator::Destroy();
    return Measurement{packets, g_received};
}
} // namespace

int
main(int argc, char* argv[])
{
    std::string output = "wsn-link-validation.csv";
    std::string exponents = "3.0,3.5";
    double fadingM = 1.0;
    double maxDistance = 80;
    double step = 2;
    uint32_t packets = 2000;
    uint32_t msduBytes = 100;
    double txPower = 0;
    double rxSensitivity = -106.58;
    uint32_t seed = 2000;

    CommandLine cmd(__FILE__);
    cmd.AddValue("output", "CSV file to write", output);
    cmd.AddValue("exponents", "comma-separated path loss exponents", exponents);
    cmd.AddValue("fadingM", "Nakagami m (1 = Rayleigh); 0 disables fading", fadingM);
    cmd.AddValue("maxDistance", "largest distance in metres", maxDistance);
    cmd.AddValue("step", "distance step in metres", step);
    cmd.AddValue("packets", "frames sent per distance", packets);
    cmd.AddValue("msduBytes", "MAC payload bytes per frame (max 114 with short addresses)",
                 msduBytes);
    cmd.AddValue("txPower", "transmit power in dBm", txPower);
    cmd.AddValue("rxSensitivity", "receiver sensitivity in dBm", rxSensitivity);
    cmd.AddValue("seed", "random seed", seed);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    std::ofstream csv(output);
    csv << "exponent,nakagami_m,distance_m,msdu_bytes,tx_power_dbm,rx_sensitivity_dbm,"
           "sent,received,psr\n";
    uint32_t run = 1;
    for (double exponent : ParseList(exponents))
    {
        for (double distance = step; distance <= maxDistance + 1e-9; distance += step)
        {
            RngSeedManager::SetRun(run++);
            Measurement m = MeasureLink(distance,
                                        exponent,
                                        fadingM,
                                        packets,
                                        msduBytes,
                                        txPower,
                                        rxSensitivity);
            csv << exponent << ',' << fadingM << ',' << distance << ',' << msduBytes << ','
                << txPower << ',' << rxSensitivity << ',' << m.sent << ',' << m.received << ','
                << static_cast<double>(m.received) / m.sent << '\n';
        }
        std::cout << "exponent " << exponent << " done\n";
    }
    csv.close();
    std::cout << "wrote " << output << "\n";
    return 0;
}
