struct JuliaHubCommercialSupportComponent <: MultiDocumenter.DropdownComponent 
    link::String
end

function MultiDocumenter.render(c::JuliaHubCommercialSupportComponent, doc, thispage, prettyurls)

    return MultiDocumenter.@htl """
    <div>
        <a href="$(c.link)" class="nav-link nav-item">
            <img src="https://info.juliahub.com/hubfs/Julia-Hub-Navigation-Logo-JuliaHub.svg" alt="JuliaHub logo - contact sales today!" style = "padding: 10px;"/>
        </a>
        <p></p>
        <a href="$(c.link)" class="nav-link nav-item">JuliaHub offers commercial support for ModelingToolkit and the SciML ecosystem.  Contact us today to discuss your needs!</a>
    </div>
    """
    
end

struct ProductsUsedComponent <: MultiDocumenter.DropdownComponent end

PRODUCTNAME_IMAGE_LINK = [
    (; product = "JuliaSim", logo = "https://juliahub.com/ui/juliasim-logo.notext.svg", link = "https://juliahub.com/products/juliasim"),
    (; product = "Pumas", logo = "https://juliahub.com/ui/Pumas%20Logomark.svg", link = "https://pumas.ai/"),
    (; product = "Cedar EDA", logo = "https://juliahub.com/ui/cedar_eda.svg", link = "https://juliahub.com/products/cedar-eda"),
    (; product = "Neuroblox", logo = "https://juliahub.com/ui/Neuroblox-logo-400-300-dark.png", link = "https://www.neuroblox.org/"),
    (; product = "Planting Space", logo = "https://planting.space/img/logo_big.svg", link = "https://planting.space/"),
]

function MultiDocumenter.render(c::ProductsUsedComponent, doc, thispage, prettyurls)
    strings = [MultiDocumenter.@htl """
    <li>
        <a href=$(product.link) class="nav-link nav-item">
            $(product.product)
        </a>
    </li>
    """ for product in PRODUCTNAME_IMAGE_LINK]

    return MultiDocumenter.@htl """
    <table>
        $strings
    </table>
    """
end


struct Link <: MultiDocumenter.DropdownComponent
    text::String
    link::String
end

function MultiDocumenter.render(c::Link, doc, thispage, prettyurls)
    return MultiDocumenter.@htl """
    <a href=$(c.link) class="nav-link nav-item">$(c.text)</a>
    """
end